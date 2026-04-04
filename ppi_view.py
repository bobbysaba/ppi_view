#!/usr/bin/env python3
from flask import Flask, Response, jsonify, render_template, request
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import argparse, csv, json, math, time, threading, glob, socket, webbrowser
import numpy as np
import netCDF4 as nc
import cmocean
from scipy.spatial import cKDTree
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default=None)   # directory containing PPI .cdf files
parser.add_argument('--sfc-dir',  type=str, default=None)   # directory containing surface obs .txt files
parser.add_argument('--vad-dir',  type=str, default=None)   # directory containing VAD .cdf files
parser.add_argument('--no-browser', action='store_true')    # suppress auto-opening the browser
parser.add_argument('--test-ppi', action='store_true',
                    help='Start in historical PPI mode using the most recent file in data-dir')
parser.add_argument('--test-vad', action='store_true',
                    help='Start in historical VAD mode using the most recent file in vad-dir')
args, _ = parser.parse_known_args()

# JSON config file sits next to this script; CLI args take precedence over it
CONFIG_FILE = Path(__file__).parent / 'ppiview.config.json'

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}

_CFG      = load_config()
# Resolve directories: CLI arg → config file → default test_data subfolder
DATA_DIR  = Path(args.data_dir or _CFG.get('data_dir', str(Path(__file__).parent / 'test_data'))).expanduser()
SFC_DIR   = Path(args.sfc_dir  or _CFG.get('sfc_dir',  str(Path(__file__).parent / 'test_data'))).expanduser()
VAD_DIR   = Path(args.vad_dir  or _CFG.get('vad_dir',  str(Path(__file__).parent / 'test_data'))).expanduser()
HTTP_PORT = int(_CFG.get('http_port', 8050))

def _sample_phase_cmap() -> list[list[int]]:
    # Sample the cmocean 'phase' colormap at 36 evenly-spaced points (one per 10° of wind direction)
    # and convert to integer RGB triplets for use in the frontend wind-direction renderer.
    cmap = cmocean.cm.phase
    return [[round(r*255), round(g*255), round(b*255)]
            for r, g, b, _ in (cmap(i / 36) for i in range(36))]

PHASE_CM = _sample_phase_cmap()

def _detect_test_date(directory: Path, pattern: str, date_re: str) -> str | None:
    # Scan the directory for files matching the ARM naming convention and extract
    # the YYYYMMDD date from the most recent file so the UI can pre-select it.
    import re
    files = sorted(glob.glob(str(directory / pattern)))
    if not files:
        return None
    m = re.search(date_re, Path(files[-1]).name)
    return m.group(1) if m else None

# Only detect test dates when the corresponding test flag is set; None means "live/today" mode.
TEST_DATE     = _detect_test_date(DATA_DIR, '*.b1.????????.*.cdf', r'\.b1\.(\d{8})\.') if args.test_ppi else None
TEST_VAD_DATE = _detect_test_date(VAD_DIR,  '*.c1.????????.*.cdf', r'\.c1\.(\d{8})\.') if args.test_vad else None

def _log(msg):
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] [ppiview] {msg}', flush=True)

app = Flask(__name__)

def find_sfc_file(date: datetime) -> Path | None:
    pattern = str(SFC_DIR / f'{date.strftime("%Y%m%d")}.txt')
    files = glob.glob(pattern)
    return Path(files[0]) if files else None

def load_surface_series(sfc_path: Path) -> list[dict]:
    series: list[dict] = []
    with open(sfc_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # gps_time is stored as a 6-digit HHMMSS integer (possibly float-encoded)
                t = str(int(float(row['gps_time']))).zfill(6)
                # Convert HHMMSS → total seconds since midnight for easy nearest-neighbor lookup
                t_s = int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
                compass = float(row['compass_dir'])
                lat = float(row['lat'])
                lon = float(row['lon'])
                # Skip rows with non-finite values (GPS dropouts, fill values, etc.)
                if math.isfinite(compass) and math.isfinite(lat) and math.isfinite(lon):
                    series.append({
                        't_s': t_s,
                        'heading': compass,
                        'lat': lat,
                        'lon': lon,
                    })
            except (ValueError, KeyError, OverflowError):
                continue
    return series

# Per-day cache so surface files are only parsed once per date
_hdg_cache: dict[str, list] = {}

def _get_hdg_series(date: datetime) -> list:
    key = date.strftime('%Y%m%d')
    if key not in _hdg_cache:
        sfc = find_sfc_file(date)
        _hdg_cache[key] = load_surface_series(sfc) if (sfc and sfc.exists()) else []
    return _hdg_cache[key]

def _heading_at(series: list, ts: float) -> float:
    """Return the compass heading (degrees) closest in time to Unix timestamp ts.
    Falls back to 0.0 (North) when no surface data is available."""
    if not series:
        return 0.0
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    t_s = dt.hour * 3600 + dt.minute * 60 + dt.second
    # Linear scan is fast enough; the series is typically a few thousand rows/day
    return min(series, key=lambda x: abs(x['t_s'] - t_s))['heading']

def _surface_sample_at(series: list, ts: float) -> dict | None:
    """Return the full surface record (heading, lat, lon) closest in time to ts."""
    if not series:
        return None
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    t_s = dt.hour * 3600 + dt.minute * 60 + dt.second
    return min(series, key=lambda x: abs(x['t_s'] - t_s))

def infer_timezone_name(lat: float, lon: float) -> str:
    """Heuristically map a lat/lon to an IANA timezone name using bounding boxes.

    This avoids a full tzdata shapefile lookup while covering the deployment regions
    (Hawaii, Alaska, Arizona no-DST zone, and the four contiguous US time zones).
    Anything outside these boxes falls back to a fixed UTC offset derived from longitude.
    """
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return 'UTC'

    # Hawaii (no DST)
    if 18 <= lat <= 23 and -161 <= lon <= -154:
        return 'Pacific/Honolulu'
    # Alaska
    if 51 <= lat <= 72 and -171 <= lon <= -129:
        return 'America/Anchorage'
    # Arizona (Mountain time but no DST)
    if 31 <= lat <= 37.5 and -115.1 <= lon <= -109.0:
        return 'America/Phoenix'

    # Contiguous US: split by longitude boundaries (approximate zone edges)
    if 24 <= lat <= 50 and -125 <= lon <= -66:
        if lon <= -114:
            return 'America/Los_Angeles'
        if lon <= -104:
            return 'America/Denver'
        if lon <= -86:
            return 'America/Chicago'
        return 'America/New_York'

    # Outside covered regions: derive a fixed-offset zone from longitude (15° per hour).
    # Note: Etc/GMT sign convention is inverted relative to UTC offset (Etc/GMT-5 = UTC+5).
    offset_hours = int(round(lon / 15.0))
    sign = '-' if offset_hours >= 0 else '+'
    return f'Etc/GMT{sign}{abs(offset_hours)}'

def local_time_info(ts: float, surface_sample: dict | None) -> dict:
    tz_name = 'UTC'
    lat = None
    lon = None
    if surface_sample:
        lat = surface_sample.get('lat')
        lon = surface_sample.get('lon')
        if lat is not None and lon is not None:
            tz_name = infer_timezone_name(lat, lon)

    utc_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    try:
        local_dt = utc_dt.astimezone(ZoneInfo(tz_name))
    except Exception:
        tz_name = 'UTC'
        local_dt = utc_dt

    return {
        'timezone': tz_name,
        'local_time': local_dt.strftime('%H:%M:%S'),
        'local_label': local_dt.strftime('%H:%M:%S %Z'),
        'lat': round(float(lat), 5) if lat is not None and math.isfinite(lat) else None,
        'lon': round(float(lon), 5) if lon is not None and math.isfinite(lon) else None,
    }

def find_vad_file(date: datetime) -> Path | None:
    pattern = str(VAD_DIR / f'*.c1.{date.strftime("%Y%m%d")}.*.cdf')
    files = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None

def load_vad(path: Path) -> dict:
    ds = nc.Dataset(str(path))
    try:
        base_time   = int(ds.variables['base_time'][:])
        time_offset = _fill_nan(ds.variables['time_offset'][:])
        timestamps  = (base_time + time_offset).tolist()
        height      = _fill_nan(ds.variables['height'][:]).tolist()
        wdir        = _to_list2d(_fill_nan(ds.variables['wdir'][:]), decimals=1)
        wspd        = _to_list2d(_fill_nan(ds.variables['wspd'][:]), decimals=2)
        w           = _to_list2d(_fill_nan(ds.variables['w'][:]),    decimals=3)
        rms         = _to_list2d(_fill_nan(ds.variables['rms'][:]),  decimals=3)
        r_sq        = _to_list2d(_fill_nan(ds.variables['r_sq'][:]), decimals=4)
        return {
            'timestamps': timestamps,
            'height_km':  height,
            'wdir':       wdir,
            'wspd':       wspd,
            'w':          w,
            'rms':        rms,
            'r_sq':       r_sq,
        }
    finally:
        ds.close()

def find_daily_file(date: datetime) -> Path | None:
    pattern = str(DATA_DIR / f'*.b1.{date.strftime("%Y%m%d")}.*.cdf')
    files = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None

def today_file() -> Path | None:
    return find_daily_file(datetime.now(timezone.utc))

def _fill_nan(arr) -> np.ndarray:
    # NetCDF masked arrays use a fill value (often 9.96921e+36) to represent missing data.
    # Convert to float64 and replace both masked values and any non-finite leftovers with NaN.
    out = np.ma.filled(arr.astype(np.float64), fill_value=np.nan)
    out[~np.isfinite(out)] = np.nan
    return out

def _to_list2d(arr: np.ndarray, decimals: int | None = None) -> list:
    # Round to reduce JSON payload size, then replace NaN/inf with None so the
    # result is valid JSON (JSON has no NaN literal).
    if decimals is not None:
        arr = np.round(arr, decimals)
    obj = np.where(np.isfinite(arr), arr, None)
    return obj.tolist()

def compute_vorticity(v: np.ndarray, az: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Estimate radial-velocity vorticity using a central-difference scheme in azimuth.

    The discrete approximation is:
        ζ ≈ (1 / r) * ∂v/∂θ  ≈  (v[i+1] - v[i-1]) / (r * Δθ)
    where θ is in radians and r is in metres.
    The first and last ray cannot be differenced, so they remain NaN.
    """
    vort = np.full_like(v, np.nan)
    if v.shape[0] >= 3:
        # Central difference in azimuth (radians); skip 2 rays at each end of the sweep
        daz = np.deg2rad(az[2:]) - np.deg2rad(az[:-2])
        with np.errstate(invalid='ignore', divide='ignore'):
            # r is in km → convert to metres so vorticity is in s⁻¹
            vort[1:-1, :] = ((v[2:, :] - v[:-2, :]) / daz[:, None]) / (r[None, :] * 1000.0)
    return vort

def compute_circulation(vort: np.ndarray, az: np.ndarray, r: np.ndarray,
                        radius_m: float = 250.0) -> np.ndarray:
    """Compute local circulation Γ by integrating vorticity flux over a disk of radius_m metres.

    For each radar gate, sum ζ * dA over all gates within radius_m, where the area element is:
        dA = r * dr * dθ  (polar area in m²)

    A cKDTree on Cartesian gate positions makes the neighbourhood query O(N log N).
    NaN vorticity gates contribute zero flux (conservative, avoids missing-data inflation).
    """
    if len(az) < 2:
        return np.full((len(az), len(r)), np.nan)
    r_m  = r * 1000.0                                    # km → m
    dr   = float(r_m[1] - r_m[0]) if len(r_m) > 1 else float(r_m[0])
    daz  = np.abs(np.gradient(np.deg2rad(az)))           # (n_rays,) per-ray angular width in radians
    area = r_m[None, :] * dr * daz[:, None]              # (n_rays, n_gates) m²: polar area element
    with np.errstate(invalid='ignore'):
        # Treat NaN vorticity as zero so missing gates don't propagate into the sum
        flux = np.where(np.isfinite(vort), vort * area, 0.0)

    # Convert polar (az, r) to Cartesian (x, y) for spatial indexing
    az_rad = np.deg2rad(az)
    x = np.outer(np.sin(az_rad), r_m)                    # (n_rays, n_gates)
    y = np.outer(np.cos(az_rad), r_m)
    pts = np.column_stack([x.ravel(), y.ravel()])

    # Query all points within radius_m of each gate and accumulate their flux
    tree      = cKDTree(pts)
    neighbors = tree.query_ball_point(pts, r=radius_m)
    flux_flat = flux.ravel()
    circ_flat = np.array([flux_flat[nb].sum() for nb in neighbors])
    return circ_flat.reshape(vort.shape)

def load_scans(path: Path, hdg: list | None = None) -> list[dict]:
    """Return a lightweight metadata list for every scan (snum) in the file.

    Each entry contains only the fields needed to populate the scan-selector UI;
    the full ray/gate arrays are not loaded here (see load_scan_data).
    """
    ds = nc.Dataset(str(path))
    try:
        # base_time is a single Unix epoch integer; time_offset is seconds since base_time
        base_time   = int(ds.variables['base_time'][:])
        time_offset = _fill_nan(ds.variables['time_offset'][:])
        # snum groups rays into discrete PPI sweeps
        snum        = ds.variables['snum'][:].data.astype(int)
        azimuth     = _fill_nan(ds.variables['azimuth'][:])
        elevation   = _fill_nan(ds.variables['elevation'][:])

        scans = []
        for s in np.unique(snum):
            mask = snum == s
            t    = base_time + time_offset[mask]
            ts   = float(np.nanmean(t))             # representative timestamp for this sweep
            h    = _heading_at(hdg, ts) if hdg else 0.0
            loc  = _surface_sample_at(hdg, ts) if hdg else None
            tloc = local_time_info(ts, loc)
            scans.append({
                'snum':      int(s),
                'timestamp': ts,
                'n_rays':    int(mask.sum()),
                'az_min':    float(round(np.nanmin(azimuth[mask]), 1)),
                'az_max':    float(round(np.nanmax(azimuth[mask]), 1)),
                'elevation': float(round(np.nanmean(elevation[mask]), 2)),
                'heading':   float(round(h, 1)),
                'local_time': tloc['local_time'],
                'local_label': tloc['local_label'],
                'timezone': tloc['timezone'],
            })
        scans.sort(key=lambda x: x['timestamp'])
        return scans
    finally:
        ds.close()

def load_scan_data(path: Path, snum_target: int, hdg: list | None = None) -> dict:
    """Load all ray/gate data for a single PPI sweep and derive computed fields.

    Returns a dict ready to JSON-serialise and send to the browser. Heavy fields
    (velocity, vorticity, circulation, etc.) are (n_rays × n_gates) 2-D lists.
    """
    ds = nc.Dataset(str(path))
    try:
        base_time   = int(ds.variables['base_time'][:])
        time_offset = _fill_nan(ds.variables['time_offset'][:])
        snum        = ds.variables['snum'][:].data.astype(int)
        mask        = snum == snum_target
        if not mask.any():
            return {}

        azimuth     = _fill_nan(ds.variables['azimuth'][mask])
        r           = _fill_nan(ds.variables['range'][:])      # range gates, shape (n_gates,)
        velocity    = _fill_nan(ds.variables['velocity'][mask])
        intensity   = _fill_nan(ds.variables['intensity'][mask])
        backscatter = _fill_nan(ds.variables['backscatter'][mask])
        elevation   = _fill_nan(ds.variables['elevation'][mask])
        t           = base_time + time_offset[mask]
        ts          = float(np.nanmean(t))

        # Drop rays with NaN azimuth, then sort by raw azimuth (lidar frame)
        valid = np.isfinite(azimuth)
        azimuth     = azimuth[valid]
        velocity    = velocity[valid]
        intensity   = intensity[valid]
        backscatter = backscatter[valid]

        order       = np.argsort(azimuth)
        azimuth     = azimuth[order]
        velocity    = velocity[order]
        intensity   = intensity[order]
        backscatter = backscatter[order]

        # Heading → geographic azimuth (kept for reference; rendering uses raw + heading)
        heading  = _heading_at(hdg, ts) if hdg else 0.0
        loc      = _surface_sample_at(hdg, ts) if hdg else None
        tloc     = local_time_info(ts, loc)
        # Rotate raw lidar azimuths into geographic (North-referenced) frame
        true_az  = (azimuth + heading) % 360.0

        # Vorticity and circulation use sorted raw azimuth so angles are monotonic
        vorticity   = compute_vorticity(velocity, azimuth, r)
        circulation = compute_circulation(vorticity, azimuth, r)

        return {
            'snum':        snum_target,
            'timestamp':   ts,
            'heading':     round(heading, 1),
            'elevation':   round(float(np.nanmean(elevation)), 2),
            'local_time':  tloc['local_time'],
            'local_label': tloc['local_label'],
            'timezone':    tloc['timezone'],
            'lat':         tloc['lat'],
            'lon':         tloc['lon'],
            'azimuth':     [round(float(v), 2) for v in azimuth],
            'true_azimuth': [round(float(v), 2) for v in true_az],
            'range_km':    [round(float(v), 4) for v in r],
            'velocity':    _to_list2d(velocity),
            'vorticity':   _to_list2d(vorticity),
            'circulation': _to_list2d(circulation, decimals=3),
            'backscatter': _to_list2d(backscatter),
            'intensity':   _to_list2d(intensity),
        }
    finally:
        ds.close()

# Shared state for the live-update pipeline; all writes must hold _cache_lock
_cache_lock  = threading.Lock()
_scan_list   = []          # most-recent scan metadata list (updated by _watcher)
_latest_snum = None        # snum of the newest scan seen
_sse_cond    = threading.Condition()  # used to wake SSE clients when a new scan arrives
_sse_seq     = 0           # monotonically incremented each time a new scan is detected

# keyed by (date_str, snum) → scan data dict
_scan_data_cache: dict[tuple, dict] = {}

# precompute state: None = idle, str = date currently being computed
_precompute_state      = None
_precompute_state_lock = threading.Lock()

def _precompute_date(date_str: str, path: Path, show_progress: bool = False) -> None:
    """Pre-populate _scan_data_cache for every scan in a day's file.

    Runs in a background thread (or blocking at startup with show_progress=True).
    Skips scans already in cache so it is safe to call multiple times.
    _precompute_state tracks the active date so the UI can show a progress indicator.
    """
    global _precompute_state
    with _precompute_state_lock:
        _precompute_state = date_str
    try:
        date = datetime.strptime(date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
        hdg  = _get_hdg_series(date)
        scans = load_scans(path, hdg)
        # Wrap in tqdm for terminal progress bar when called at startup
        it = tqdm(scans, desc=f'Computing circulation ({date_str})', unit='scan') \
             if show_progress else scans
        for meta in it:
            key = (date_str, meta['snum'])
            with _cache_lock:
                already = key in _scan_data_cache
            if not already:
                data = load_scan_data(path, meta['snum'], hdg)
                with _cache_lock:
                    _scan_data_cache[key] = data
    finally:
        # Always clear state so the UI stops showing the progress indicator
        with _precompute_state_lock:
            _precompute_state = None

def _watcher():
    """Background thread that polls today's data file every 15 seconds.

    When a new scan number appears, it increments _sse_seq and notifies all
    waiting SSE connections so the browser receives a push event immediately.
    """
    global _scan_list, _latest_snum, _sse_seq
    while True:
        try:
            now  = datetime.now(timezone.utc)
            path = today_file()
            if path and path.exists():
                hdg   = _get_hdg_series(now)
                scans = load_scans(path, hdg)
                if scans:
                    newest = scans[-1]['snum']
                    with _cache_lock:
                        changed      = newest != _latest_snum
                        _scan_list   = scans
                        _latest_snum = newest
                    if changed:
                        # Wake all SSE clients so they can push the new snum to the browser
                        with _sse_cond:
                            _sse_seq += 1
                            _sse_cond.notify_all()
                        _log(f'New scan detected: snum={newest}')
        except Exception as e:
            _log(f'Watcher error: {e}')
        time.sleep(15)

# Start watcher as a daemon so it exits automatically when the main process exits
threading.Thread(target=_watcher, daemon=True).start()

def _parse_date(date_str: str | None) -> datetime:
    if date_str:
        return datetime.strptime(date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)

@app.route('/')
def index():
    return render_template('index.html', test_date=TEST_DATE)

@app.route('/scans')
def scans():
    date_str = request.args.get('date')
    date     = _parse_date(date_str)
    path     = find_daily_file(date)

    if not path or not path.exists():
        return jsonify({'scans': [], 'file': None})

    if not date_str:                       # today — serve the in-memory list without re-reading disk
        with _cache_lock:
            if _scan_list:
                return jsonify({'scans': list(_scan_list), 'file': path.name})

    try:
        hdg = _get_hdg_series(date)
        return jsonify({'scans': load_scans(path, hdg), 'file': path.name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scan/<int:snum>')
def scan(snum):
    date_str = request.args.get('date')
    date     = _parse_date(date_str)
    key_date = date.strftime('%Y%m%d')
    path     = find_daily_file(date)

    if not path or not path.exists():
        return jsonify({'error': 'file not found'}), 404

    try:
        with _cache_lock:
            data = _scan_data_cache.get((key_date, snum))
        if data is None:
            hdg  = _get_hdg_series(date)
            data = load_scan_data(path, snum, hdg)
            with _cache_lock:
                _scan_data_cache[(key_date, snum)] = data
        if not data:
            return jsonify({'error': 'scan not found'}), 404
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/vad')
def vad_page():
    return render_template('vad.html', test_vad_date=TEST_VAD_DATE, phase_cm=PHASE_CM)

@app.route('/vad/data')
def vad_data():
    date_str = request.args.get('date')
    date     = _parse_date(date_str)
    path     = find_vad_file(date)
    if not path or not path.exists():
        return jsonify({'error': 'VAD file not found'}), 404
    try:
        return jsonify(load_vad(path))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/precompute_status')
def precompute_status():
    with _precompute_state_lock:
        state = _precompute_state
    return jsonify({'computing': state is not None, 'date': state})

@app.route('/precompute', methods=['POST'])
def precompute():
    date_str = request.json.get('date')
    if not date_str:
        return jsonify({'error': 'date required'}), 400
    date = _parse_date(date_str)
    path = find_daily_file(date)
    if not path or not path.exists():
        return jsonify({'error': 'file not found'}), 404
    threading.Thread(
        target=_precompute_date, args=(date_str, path, False), daemon=True
    ).start()
    return jsonify({'started': True})

@app.route('/stream')
def stream():
    """Server-Sent Events endpoint — pushes a JSON object whenever a new scan is detected.

    The browser opens one persistent connection here; _watcher notifies _sse_cond when
    _sse_seq increments. A keep-alive comment is sent every ~20 s to prevent proxy timeouts.
    X-Accel-Buffering: no disables nginx's response buffering so events arrive immediately.
    """
    def generate():
        # Snapshot the current sequence number so we only send genuinely new events
        with _sse_cond:
            last_seq = _sse_seq
        try:
            while True:
                with _sse_cond:
                    # Block until _sse_seq changes or 20-second timeout fires
                    changed = _sse_cond.wait_for(lambda: _sse_seq != last_seq, timeout=20)
                    if changed:
                        with _cache_lock:
                            snum = _latest_snum
                        last_seq = _sse_seq
                        yield f'data: {json.dumps({"snum": snum})}\n\n'
                    else:
                        # SSE comment line — not dispatched as an event but keeps the TCP connection alive
                        yield ': keep-alive\n\n'
        except GeneratorExit:
            pass  # client disconnected; generator cleanup is automatic

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

def get_local_ip():
    # Trick: open a UDP socket to a public address without actually sending data;
    # the OS fills in the source address, revealing the machine's LAN IP.
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        if s: s.close()

if __name__ == '__main__':
    _log(f'Data directory : {DATA_DIR}')
    _log(f'Surface obs dir: {SFC_DIR}')

    # Precompute circulation for the startup date (blocking, PPI mode only)
    if not args.test_vad:
        startup_date = TEST_DATE or datetime.now(timezone.utc).strftime('%Y%m%d')
        startup_path = find_daily_file(
            datetime.strptime(startup_date, '%Y%m%d').replace(tzinfo=timezone.utc)
        )
        if startup_path and startup_path.exists():
            _precompute_date(startup_date, startup_path, show_progress=True)
        else:
            _log('No data file found for startup date — skipping precomputation')

    ip  = get_local_ip()
    url = f'http://{ip}:{HTTP_PORT}'
    if args.test_vad:
        url += '/vad'
    _log(f'Starting PPI Viewer — {url}')
    if not args.no_browser:
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=False, threaded=True)
