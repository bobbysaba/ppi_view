# ppi_view

Real-time PPI (Plan Position Indicator) viewer for CLAMPS Doppler lidar data. Serves a browser-based interface that displays radial velocity, vorticity, circulation, backscatter, and intensity from daily NetCDF files, with optional surface obs overlay for truck heading and GPS position.

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
| `data_dir` | Path to daily lidar `.cdf` files |
| `sfc_dir` | Path to daily surface obs `.txt` files |
| `http_port` | Port to serve on (default: `8050`) |

Data directories can also be passed as CLI flags, which take precedence over the config file.

## Usage

```bash
# Live mode — watches today's data file and updates in real time
python ppi_view.py

# Override data directories at runtime
python ppi_view.py --data-dir /path/to/data --sfc-dir /path/to/sfc

# Historical/test mode — loads the most recent file in data-dir
python ppi_view.py --test

# Suppress automatic browser launch
python ppi_view.py --no-browser
```

The viewer opens automatically at `http://<host-ip>:8050`.

## Data Format

**Lidar files** — NetCDF (`.cdf`), named `*.b1.YYYYMMDD.HHMMSS.cdf`, containing variables: `base_time`, `time_offset`, `snum`, `azimuth`, `elevation`, `range`, `velocity`, `intensity`, `backscatter`.

**Surface obs files** — CSV (`.txt`), named `YYYYMMDD.txt`, with columns: `gps_time`, `compass_dir`, `lat`, `lon`.
