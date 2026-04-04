# vad.html Build Plan

## Layout

Two-column layout inside a flex container, sitting below a shared header/controls bar:

```
┌─────────────────────────────────────────────────────┐
│  Header: title, date picker, status                 │
├───────────────────────┬─────────────────────────────┤
│                       │  wspd time-height           │
│   Hodograph           │  (time × height heatmap)    │
│   (height-colormapped │─────────────────────────────┤
│    with colorbar)     │  wdir time-height           │
│                       │  (time × height heatmap)    │
└───────────────────────┴─────────────────────────────┘
```

- Left column: hodograph canvas + height colorbar + legend
- Right column: two stacked time-height canvases (wspd on top, wdir on bottom)
- Both columns share the same height, filling the viewport

---

## Components

### Header / Controls
- Title: `NSSL Lidar VAD Viewer`
- Date picker (same style as PPI viewer)
- r² threshold slider — masks gates where `r_sq` is below the threshold before rendering
- Status dot (static for now, live-ready later)

### Hodograph (left canvas)
- Rendered via Canvas 2D API
- X axis = U (eastward wind), Y axis = V (northward wind)
- Each point: `U = wspd * sin(wdir_rad)`, `V = wspd * cos(wdir_rad)`
- Points connected by line segments, colored by height using a sequential colormap (e.g. viridis-like: purple → blue → green → yellow)
- Height colorbar drawn below the canvas with min/max labels (km AGL)
- Concentric speed rings drawn at 5, 10, 15, 20 m/s with labels
- Cardinal axis labels (N/S/E/W or +U/−U/+V/−V)
- Title shows selected timestamp (local label from `timestamps` array)
- Updates when time cursor changes

### Time-height plots (right column, two stacked canvases)
- X axis = time (52 steps), Y axis = height (400 levels, clipped to a sensible max e.g. 4 km)
- Top canvas: wspd — sequential colormap (0 → ~25 m/s)
- Bottom canvas: wdir — cyclic colormap (0–360°, wraps cleanly)
- Vertical time cursor line drawn on both canvases at the selected time index
- Click or scrub on either canvas to select a time step → updates hodograph
- Axis labels: time as HH:MM UTC on x, height in km on y
- Colorbar drawn below each canvas

### Time scrubber
- Range input below the two-column layout (same style as PPI scrubber)
- Shows current timestamp label
- Prev / Next step buttons flanking it

---

## Data Flow

1. On load, fetch `/vad/data?date=YYYYMMDD` (date from `TEST_VAD_DATE` or date picker)
2. Store full payload: `timestamps`, `height_km`, `wdir`, `wspd`, `rms`, `r_sq`
3. Apply r² mask: set gates to `null` where `r_sq[t][h] < threshold`
4. Clip height axis to a configurable max (default 4 km) to avoid empty upper levels
5. On time index change: redraw hodograph + redraw cursor line on both time-height canvases

---

## Colormaps

| Field | Colormap | Range |
|---|---|---|
| Hodograph height | viridis-like sequential | 0 – max height km |
| wspd | sequential (white → blue → purple) | 0 – 25 m/s |
| wdir | cyclic HSV-like | 0 – 360° |

---

## Style

- Matches `index.html` exactly: Catppuccin Mocha palette, `ui-monospace` font, same card/header/button classes
- All rendering via Canvas 2D (no external charting libraries)
- `TEST_VAD_DATE` injected from Flask via `{{ test_vad_date | tojson }}`
