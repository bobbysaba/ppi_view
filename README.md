# NSSL DL Truck Data Viewer

Real-time browser-based viewer for NSSL Doppler lidar data collected from the DL Truck platform. Displays PPI, RHI, and VAD scan types from daily NetCDF files, with optional surface obs overlay for platform heading and GPS position.

## Features

**Dashboard**
- Unified landing page with live status cards for PPI, RHI, and VAD
- Per-card last-scan age (color-coded green / orange / red), scan count, and UTC time range for the day
- Hourly scan-count sparkline for PPI
- Platform position and heading (lat, lon, heading) sourced from co-located surface obs
- Data gap alert banner when any scan type has not updated in > 10 minutes
- Date picker pre-selects a historical date across all viewers before opening one

**PPI viewer** (`/ppi`)
- Displays radial velocity, backscatter, and intensity from Doppler lidar PPI sweeps
- Computes and renders **vorticity** (central-difference ∂v/∂θ) and **local circulation** (area-integrated vorticity within a 250 m radius) as derived products
- Rotates raw lidar azimuths into a geographic (North-referenced) frame using platform heading from surface obs
- Live mode polls today's data file every 15 seconds and pushes new scans to the browser via **Server-Sent Events**
- Background **precomputation** calculates circulation for all scans at startup so switching between sweeps is instant
- Dual-panel layout with independent field selection, shared zoom/pan, and circle-drag zoom
- Cursor readout of range, bearing, and field values at the pointer position

**RHI viewer** (`/rhi`)
- Cartesian cross-section rendering (x = horizontal range, y = height AGL)
- Displays radial velocity, backscatter, and intensity
- Overlays: height grid lines (AGL), range rings, elevation angle grid, ground baseline
- Same dual-panel layout, zoom/pan, and cursor readout as the PPI viewer

**VAD viewer** (`/vad`)
- Time-height plots of wind speed, wind direction, vertical velocity (w), RMS error, and R²
- **Hodograph** rendering of the horizontal wind profile at a selected time step
- Wind direction encoded with the cmocean `phase` colormap

**Platform support**
- Infers local timezone from GPS position (covers Hawaii, Alaska, Arizona, and the four contiguous US zones)
- Exposes the LAN IP at startup so the viewer is accessible from any device on the same network
- All viewers support a `?date=YYYYMMDD` URL parameter for direct deep-linking to a historical date

## Requirements

- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda)

## Setup

```bash
conda env create -f environment.yml
conda activate lid_viewer
```

## Configuration

Copy the example config and edit paths to match your data:

```bash
cp ppiview.config.example.json ppiview.config.json
```

| Key | Description |
|---|---|
| `data_dir` | Path to daily PPI lidar `.cdf` files |
| `sfc_dir` | Path to daily surface obs `.txt` files |
| `vad_dir` | Path to daily VAD lidar `.cdf` files |
| `rhi_dir` | Path to daily RHI lidar `.cdf` files |
| `http_port` | Port to serve on (default: `8050`) |

Data directories can also be passed as CLI flags, which take precedence over the config file.

## Usage

```bash
# Live mode — watches today's data files and updates in real time
python lid_viewer.py

# Override data directories at runtime
python lid_viewer.py --data-dir /path/to/ppi --sfc-dir /path/to/sfc \
                   --vad-dir /path/to/vad --rhi-dir /path/to/rhi

# Test/historical mode — loads the most recent file for each scan type
# and opens the dashboard with all viewers pre-loaded on that date
python lid_viewer.py --test

# Suppress automatic browser launch
python lid_viewer.py --no-browser
```

The dashboard opens at `http://<host-ip>:8050`. Individual viewers are at `/ppi`, `/rhi`, and `/vad`.

## Data Format

**PPI / RHI lidar files** — NetCDF (`.cdf`), named `*.b1.YYYYMMDD.HHMMSS.cdf`, containing variables: `base_time`, `time_offset`, `snum`, `azimuth`, `elevation`, `range`, `velocity`, `intensity`, `backscatter`. PPI and RHI files live in separate directories; scan type is determined by the configured directory, not the filename.

**VAD lidar files** — NetCDF (`.cdf`), named `*.c1.YYYYMMDD.HHMMSS.cdf`, containing variables: `base_time`, `time_offset`, `height`, `wdir`, `wspd`, `w`, `rms`, `r_sq`.

**Surface obs files** — CSV (`.txt`), named `YYYYMMDD.txt`, with columns: `gps_time`, `compass_dir`, `lat`, `lon`.
