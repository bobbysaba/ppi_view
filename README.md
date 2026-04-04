# ppi_view

Real-time PPI (Plan Position Indicator) and VAD viewer for NSSL Doppler lidar data. Serves a browser-based interface that displays radial velocity, vorticity, circulation, backscatter, and intensity from daily NetCDF files, with optional surface obs overlay for truck heading and GPS position. A separate VAD viewer displays hodograph and time-height wind speed/direction plots.

## Features

**PPI viewer**
- Displays radial velocity, backscatter, and intensity from Doppler lidar PPI sweeps
- Computes and renders **vorticity** (central-difference ∂v/∂θ) and **local circulation** (area-integrated vorticity within a 250 m radius) as derived products
- Rotates raw lidar azimuths into a geographic (North-referenced) frame using platform heading from co-located surface obs
- Live mode polls today's data file every 15 seconds and pushes new scans to the browser via **Server-Sent Events** — no manual refresh needed
- Background **precomputation** calculates derived fields for all scans at startup so switching between sweeps is instant

**VAD viewer**
- Time-height plots of wind speed, wind direction, vertical velocity (w), RMS error, and R²
- **Hodograph** rendering of the horizontal wind profile at a selected time step
- Wind direction encoded with the cmocean `phase` colormap for intuitive rotational display

**Platform support**
- Infers local timezone from GPS position (covers Hawaii, Alaska, Arizona, and the four contiguous US zones) so all scan timestamps display in local time
- Exposes the LAN IP at startup so the viewer is immediately accessible from any device on the same network

## Requirements

- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda)

## Setup

```bash
conda env create -f environment.yml
conda activate ppi_view
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
| `http_port` | Port to serve on (default: `8050`) |

Data directories can also be passed as CLI flags, which take precedence over the config file.

## Usage

```bash
# Live mode — watches today's data file and updates in real time
python ppi_view.py

# Override data directories at runtime
python ppi_view.py --data-dir /path/to/data --sfc-dir /path/to/sfc --vad-dir /path/to/vad

# Historical/test mode — loads the most recent PPI file in data-dir
python ppi_view.py --test-ppi

# Historical/test mode — loads the most recent VAD file in vad-dir
python ppi_view.py --test-vad

# Suppress automatic browser launch
python ppi_view.py --no-browser
```

The PPI viewer opens at `http://<host-ip>:8050` and the VAD viewer at `http://<host-ip>:8050/vad`.

## Data Format

**PPI lidar files** — NetCDF (`.cdf`), named `*.b1.YYYYMMDD.HHMMSS.cdf`, containing variables: `base_time`, `time_offset`, `snum`, `azimuth`, `elevation`, `range`, `velocity`, `intensity`, `backscatter`.

**VAD lidar files** — NetCDF (`.cdf`), named `*.c1.YYYYMMDD.HHMMSS.cdf`, containing variables: `base_time`, `time_offset`, `height`, `wdir`, `wspd`, `w`, `rms`, `r_sq`.

**Surface obs files** — CSV (`.txt`), named `YYYYMMDD.txt`, with columns: `gps_time`, `compass_dir`, `lat`, `lon`.
