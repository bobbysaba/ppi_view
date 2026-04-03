#!/usr/bin/env python3
"""
ppiview — real-time PPI viewer for CLAMPS Doppler lidar data.

Usage:  python ppiview.py [--data-dir /path/to/data]
Access: http://<host-ip>:8050
"""

from flask import Flask, Response, jsonify, render_template, request
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, time, threading, glob, socket, webbrowser
import numpy as np
import netCDF4 as nc

# ── Args / Config ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default=None)
parser.add_argument('--no-browser', action='store_true')
args, _ = parser.parse_known_args()

CONFIG_FILE = Path(__file__).parent / 'ppiview.config.json'

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}

_CFG = load_config()
DATA_DIR  = Path(args.data_dir or _CFG.get('data_dir', str(Path(__file__).parent / 'test_data'))).expanduser()
HTTP_PORT = int(_CFG.get('http_port', 8050))

def _log(msg):
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] [ppiview] {msg}', flush=True)

app = Flask(__name__)

# ── NetCDF helpers ────────────────────────────────────────────────────────────
def find_daily_file(date: datetime) -> Path | None:
    pattern = str(DATA_DIR / f'*.b1.{date.strftime("%Y%m%d")}.*.cdf')
    files = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None

def today_file() -> Path | None:
    return find_daily_file(datetime.now(timezone.utc))

def load_scans(path: Path) -> list[dict]:
    """Return a list of scan metadata dicts (no field data) sorted by time."""
    ds = nc.Dataset(str(path))
    try:
        base_time    = int(ds.variables['base_time'][:])
        time_offset  = ds.variables['time_offset'][:].data
        snum         = ds.variables['snum'][:].data.astype(int)
        azimuth      = ds.variables['azimuth'][:].data
        elevation    = ds.variables['elevation'][:].data

        scans = []
        for s in np.unique(snum):
            mask = snum == s
            t    = base_time + time_offset[mask]
            scans.append({
                'snum':      int(s),
                'timestamp': float(t.mean()),
                'n_rays':    int(mask.sum()),
                'az_min':    float(azimuth[mask].min()),
                'az_max':    float(azimuth[mask].max()),
                'elevation': float(np.round(elevation[mask].mean(), 2)),
            })
        scans.sort(key=lambda x: x['timestamp'])
        return scans
    finally:
        ds.close()

def load_scan_data(path: Path, snum_target: int) -> dict:
    """Return full polar data for one scan."""
    ds = nc.Dataset(str(path))
    try:
        base_time   = int(ds.variables['base_time'][:])
        time_offset = ds.variables['time_offset'][:].data
        snum        = ds.variables['snum'][:].data.astype(int)
        mask        = snum == snum_target

        if not mask.any():
            return {}

        azimuth     = ds.variables['azimuth'][mask].data
        r           = ds.variables['range'][:].data
        velocity    = ds.variables['velocity'][mask].data
        intensity   = ds.variables['intensity'][mask].data
        backscatter = ds.variables['backscatter'][mask].data
        t           = base_time + time_offset[mask]

        # Sort rays by azimuth for clean rendering
        order       = np.argsort(azimuth)
        azimuth     = azimuth[order]
        velocity    = velocity[order]
        intensity   = intensity[order]
        backscatter = backscatter[order]

        return {
            'snum':       snum_target,
            'timestamp':  float(t.mean()),
            'azimuth':    azimuth.tolist(),
            'range_km':   r.tolist(),
            'velocity':   velocity.tolist(),
            'intensity':  intensity.tolist(),
            'backscatter': backscatter.tolist(),
        }
    finally:
        ds.close()

# ── Scan cache ────────────────────────────────────────────────────────────────
_cache_lock  = threading.Lock()
_scan_list   = []          # list of scan metadata dicts
_latest_snum = None        # snum of the most recently detected scan
_sse_cond    = threading.Condition()
_sse_seq     = 0           # incremented whenever a new scan is detected

def _watcher():
    """Poll the daily file every 15 s; update scan list and notify SSE clients."""
    global _scan_list, _latest_snum, _sse_seq
    while True:
        try:
            path = today_file()
            if path and path.exists():
                scans = load_scans(path)
                if scans:
                    newest = scans[-1]['snum']
                    with _cache_lock:
                        changed = newest != _latest_snum
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
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scans')
def scans():
    """Return scan list for today's file (or a requested date)."""
    date_str = request.args.get('date')
    if date_str:
        try:
            date = datetime.strptime(date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
        except ValueError:
            return jsonify({'error': 'invalid date'}), 400
        path = find_daily_file(date)
    else:
        path = today_file()

    if not path or not path.exists():
        return jsonify({'scans': [], 'file': None})

    with _cache_lock:
        cached = list(_scan_list) if not date_str else None

    if cached is not None:
        return jsonify({'scans': cached, 'file': path.name})

    try:
        return jsonify({'scans': load_scans(path), 'file': path.name})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scan/<int:snum>')
def scan(snum):
    """Return full polar data for one scan."""
    date_str = request.args.get('date')
    if date_str:
        try:
            date = datetime.strptime(date_str, '%Y%m%d').replace(tzinfo=timezone.utc)
        except ValueError:
            return jsonify({'error': 'invalid date'}), 400
        path = find_daily_file(date)
    else:
        path = today_file()

    if not path or not path.exists():
        return jsonify({'error': 'file not found'}), 404

    try:
        data = load_scan_data(path, snum)
        if not data:
            return jsonify({'error': 'scan not found'}), 404
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stream')
def stream():
    """SSE endpoint — notifies clients when a new scan is available."""
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
    _log(f'Data directory: {DATA_DIR}')
    ip  = get_local_ip()
    url = f'http://{ip}:{HTTP_PORT}'
    _log(f'Starting PPI Viewer — {url}')
    if not args.no_browser:
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass
    app.run(host='0.0.0.0', port=HTTP_PORT, debug=False, threaded=True)
