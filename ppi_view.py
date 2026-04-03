#!/usr/bin/env python3
"""
ppiview — real-time PPI viewer for CLAMPS Doppler lidar data.

Usage:  python ppi_view.py [--data-dir /path/to/data] [--sfc-dir /path/to/sfc]
Access: http://<host-ip>:8050
"""

from flask import Flask, Response, jsonify, render_template, request
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import argparse, csv, json, math, time, threading, glob, socket, webbrowser
import numpy as np
import netCDF4 as nc

# ── Args / Config ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default=None)
parser.add_argument('--sfc-dir',  type=str, default=None)
parser.add_argument('--no-browser', action='store_true')
parser.add_argument('--test', action='store_true',
                    help='Start in historical mode using the most recent file in data-dir')
args, _ = parser.parse_known_args()

CONFIG_FILE = Path(__file__).parent / 'ppiview.config.json'

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}

_CFG      = load_config()
DATA_DIR  = Path(args.data_dir or _CFG.get('data_dir', str(Path(__file__).parent / 'test_data'))).expanduser()
SFC_DIR   = Path(args.sfc_dir  or _CFG.get('sfc_dir',  str(Path(__file__).parent / 'test_data'))).expanduser()
HTTP_PORT = int(_CFG.get('http_port', 8050))

# In --test mode, find the most recent CDF date in DATA_DIR and pre-select it
def _detect_test_date() -> str | None:
    import re
    files = sorted(glob.glob(str(DATA_DIR / '*.b1.????????.*.cdf')))
    if not files:
        return None
    m = re.search(r'\.b1\.(\d{8})\.', Path(files[-1]).name)
    return m.group(1) if m else None

TEST_DATE = _detect_test_date() if args.test else None

def _log(msg):
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] [ppiview] {msg}', flush=True)

app = Flask(__name__)

# ── Surface obs / heading ──────────────────────────────────────────────────────
def find_sfc_file(date: datetime) -> Path | None:
    pattern = str(SFC_DIR / f'{date.strftime("%Y%m%d")}.txt')
    files = glob.glob(pattern)
    return Path(files[0]) if files else None

def load_surface_series(sfc_path: Path) -> list[dict]:
    """Return list of surface samples with UTC seconds, heading, and location."""
    series: list[dict] = []
    with open(sfc_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                t = str(int(float(row['gps_time']))).zfill(6)
                t_s = int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
                compass = float(row['compass_dir'])
                lat = float(row['lat'])
                lon = float(row['lon'])
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

_hdg_cache: dict[str, list] = {}

def _get_hdg_series(date: datetime) -> list:
    key = date.strftime('%Y%m%d')
    if key not in _hdg_cache:
        sfc = find_sfc_file(date)
        _hdg_cache[key] = load_surface_series(sfc) if (sfc and sfc.exists()) else []
    return _hdg_cache[key]

def _heading_at(series: list, ts: float) -> float:
    """Nearest-neighbour heading lookup by epoch timestamp."""
    if not series:
        return 0.0
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    t_s = dt.hour * 3600 + dt.minute * 60 + dt.second
    return min(series, key=lambda x: abs(x['t_s'] - t_s))['heading']

def _surface_sample_at(series: list, ts: float) -> dict | None:
    if not series:
        return None
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    t_s = dt.hour * 3600 + dt.minute * 60 + dt.second
    return min(series, key=lambda x: abs(x['t_s'] - t_s))

def infer_timezone_name(lat: float, lon: float) -> str:
    """Approximate IANA time zone from truck coordinates.

    This app runs without a timezone polygon database, so we use bounded
    heuristics for the US where the truck normally operates.
    """
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return 'UTC'

    if 18 <= lat <= 23 and -161 <= lon <= -154:
        return 'Pacific/Honolulu'
    if 51 <= lat <= 72 and -171 <= lon <= -129:
        return 'America/Anchorage'
    if 31 <= lat <= 37.5 and -115.1 <= lon <= -109.0:
        return 'America/Phoenix'

    if 24 <= lat <= 50 and -125 <= lon <= -66:
        if lon <= -114:
            return 'America/Los_Angeles'
        if lon <= -104:
            return 'America/Denver'
        if lon <= -86:
            return 'America/Chicago'
        return 'America/New_York'

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

# ── NetCDF helpers ────────────────────────────────────────────────────────────
def find_daily_file(date: datetime) -> Path | None:
    pattern = str(DATA_DIR / f'*.b1.{date.strftime("%Y%m%d")}.*.cdf')
    files = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None

def today_file() -> Path | None:
    return find_daily_file(datetime.now(timezone.utc))

def _fill_nan(arr) -> np.ndarray:
    """Masked array → float64, fill values and inf → NaN."""
    out = np.ma.filled(arr.astype(np.float64), fill_value=np.nan)
    out[~np.isfinite(out)] = np.nan
    return out

def _to_list2d(arr: np.ndarray) -> list:
    """2D float array → nested Python list, NaN/inf → None (JSON null)."""
    obj = np.where(np.isfinite(arr), arr, None)   # object array; None → JSON null
    return obj.tolist()

def compute_vorticity(v: np.ndarray, az: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Centred finite-difference radial vorticity (s⁻¹).
    v : (n_rays, n_range)  float64, NaN where missing
    az: (n_rays,)          degrees, sorted ascending (lidar frame)
    r : (n_range,)         km
    """
    vort = np.full_like(v, np.nan)
    if v.shape[0] >= 3:
        daz = np.deg2rad(az[2:]) - np.deg2rad(az[:-2])
        with np.errstate(invalid='ignore', divide='ignore'):
            vort[1:-1, :] = ((v[2:, :] - v[:-2, :]) / daz[:, None]) / (r[None, :] * 1000.0)
    return vort

def load_scans(path: Path, hdg: list | None = None) -> list[dict]:
    """Return scan metadata list sorted by time."""
    ds = nc.Dataset(str(path))
    try:
        base_time   = int(ds.variables['base_time'][:])
        time_offset = _fill_nan(ds.variables['time_offset'][:])
        snum        = ds.variables['snum'][:].data.astype(int)
        azimuth     = _fill_nan(ds.variables['azimuth'][:])
        elevation   = _fill_nan(ds.variables['elevation'][:])

        scans = []
        for s in np.unique(snum):
            mask = snum == s
            t    = base_time + time_offset[mask]
            ts   = float(np.nanmean(t))
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
    """Return full polar data for one scan."""
    ds = nc.Dataset(str(path))
    try:
        base_time   = int(ds.variables['base_time'][:])
        time_offset = _fill_nan(ds.variables['time_offset'][:])
        snum        = ds.variables['snum'][:].data.astype(int)
        mask        = snum == snum_target
        if not mask.any():
            return {}

        azimuth     = _fill_nan(ds.variables['azimuth'][mask])
        r           = _fill_nan(ds.variables['range'][:])
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
        true_az  = (azimuth + heading) % 360.0

        # Vorticity uses sorted raw azimuth so finite-differences are physically meaningful
        vorticity = compute_vorticity(velocity, azimuth, r)

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
            'backscatter': _to_list2d(backscatter),
            'intensity':   _to_list2d(intensity),
        }
    finally:
        ds.close()

# ── Scan cache ────────────────────────────────────────────────────────────────
_cache_lock  = threading.Lock()
_scan_list   = []
_latest_snum = None
_sse_cond    = threading.Condition()
_sse_seq     = 0

def _watcher():
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
                        with _sse_cond:
                            _sse_seq += 1
                            _sse_cond.notify_all()
                        _log(f'New scan detected: snum={newest}')
        except Exception as e:
            _log(f'Watcher error: {e}')
        time.sleep(15)

threading.Thread(target=_watcher, daemon=True).start()

# ── Routes ────────────────────────────────────────────────────────────────────
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

    if not date_str:                       # today — try in-memory cache first
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
    path     = find_daily_file(date)

    if not path or not path.exists():
        return jsonify({'error': 'file not found'}), 404

    try:
        hdg  = _get_hdg_series(date)
        data = load_scan_data(path, snum, hdg)
        if not data:
            return jsonify({'error': 'scan not found'}), 404
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stream')
def stream():
    def generate():
        with _sse_cond:
            last_seq = _sse_seq
        try:
            while True:
                with _sse_cond:
                    changed = _sse_cond.wait_for(lambda: _sse_seq != last_seq, timeout=20)
                    if changed:
                        with _cache_lock:
                            snum = _latest_snum
                        last_seq = _sse_seq
                        yield f'data: {json.dumps({"snum": snum})}\n\n'
                    else:
                        yield ': keep-alive\n\n'
        except GeneratorExit:
            pass

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# ── Main ──────────────────────────────────────────────────────────────────────
def get_local_ip():
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
    ip  = get_local_ip()
    url = f'http://{ip}:{HTTP_PORT}'
    _log(f'Starting PPI Viewer — {url}')
    if not args.no_browser:
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=False, threaded=True)
