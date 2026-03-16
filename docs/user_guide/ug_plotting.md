# Plotting

## Overview

The plotting module provides tools to visualize power line sections in 2D and 3D using [Plotly](https://plotly.com/python/).

The main entry point is the `PlotEngine` class, which is built from a `BalanceEngine` and provides methods to generate 3D and 2D views of the line, including cables, supports and insulators.

## PlotEngine

### Creating a PlotEngine

A `PlotEngine` is created from a `BalanceEngine` using the `builder_from_balance_engine` factory method:

```python
import plotly.graph_objects as go
from mechaphlowers.plotting.plot import PlotEngine

# Assuming balance_engine is an existing BalanceEngine object
plt_engine = PlotEngine.builder_from_balance_engine(balance_engine)
```

!!! important
    The `BalanceEngine` must have been solved (adjustment and change of state) before creating a `PlotEngine`.

### Reactive updates

If the `BalanceEngine` attributes change after creating the `PlotEngine` (e.g., new weather conditions), you can regenerate the plot engine:

```python
balance_engine.solve_change_state(wind_pressure=500)
plt_engine = plt_engine.generate_reset()
```

### Getting points data

The `get_points_for_plot` method returns the computed coordinates of all plotted objects as three `Points` objects (spans, supports, insulators):

```python
span, supports, insulators = plt_engine.get_points_for_plot()
```

These `Points` objects can be used for custom analysis, or passed to `compute_aspect_ratio` (see below).

For 2D projections, set `project=True` and specify a `frame_index`:

```python
span, supports, insulators = plt_engine.get_points_for_plot(
    project=True, frame_index=0
)
```

## 3D visualization

### preview_line3d

Use `preview_line3d` to add 3D traces (cables, supports, insulators) to a Plotly figure:

```python
fig = go.Figure()
plt_engine.preview_line3d(fig)
fig.show()
```

The `view` argument controls the layout mode:

- `"full"` (default): uses Plotly's `aspectmode="data"` to respect real-world scale.
- `"analysis"`: uses `aspectmode="manual"` with a compact view, better for inspecting shapes.

```python
fig = go.Figure()
plt_engine.preview_line3d(fig, view="analysis")
fig.show()
```

### Overlaying multiple states

You can overlay multiple weather or load states on the same figure by calling `preview_line3d` several times. Use the `mode` argument to visually distinguish them:

```python
fig = go.Figure()

# First state: default conditions (shown as background)
plt_engine.preview_line3d(fig, mode="background")

# Second state: after weather change
balance_engine.solve_change_state(wind_pressure=500)
plt_engine = plt_engine.generate_reset()
plt_engine.preview_line3d(fig, mode="main")

fig.show()
```

- `"main"` (default): standard solid traces.
- `"background"`: dotted traces for secondary/reference states.

## 2D visualization

### preview_line2d

Use `preview_line2d` to add 2D traces to a Plotly figure. The 3D coordinates are projected into a support-local frame.

Two views are available:

- `"profile"` (default): X axis is along the span direction, Y axis is altitude. The Y axis auto-ranges to show all data.
- `"line"`: X axis is the transverse direction (perpendicular to span), Y axis is altitude. The scale is anchored to preserve real proportions.

```python
fig_profile = go.Figure()
plt_engine.preview_line2d(fig_profile, view="profile", frame_index=0)
fig_profile.show()

fig_line = go.Figure()
plt_engine.preview_line2d(fig_line, view="line", frame_index=0)
fig_line.show()
```

The `frame_index` argument selects which support is used as the origin for the 2D projection. It must be between $0$ and $N_{supports} - 1$.

## Aspect ratio and layout

### The problem

In 3D power line plots, the horizontal extent (X/Y) of a section is typically much larger than the vertical extent (Z). Plotly's default `aspectmode="data"` preserves real-world proportions, but the altitude differences may become hard to see.

The `compute_aspect_ratio` function solves this by computing the true data ranges and allowing you to exaggerate specific axes.

### compute_aspect_ratio

This standalone function takes the `Points` objects from `get_points_for_plot` and computes a Plotly-compatible aspect ratio dictionary:

```python
from mechaphlowers.plotting import compute_aspect_ratio

span, supports, insulators = plt_engine.get_points_for_plot()

# Default: normalized ratios reflecting the real data ranges
aspect = compute_aspect_ratio(span, supports, insulators)
# e.g. {'x': 1.0, 'y': 0.45, 'z': 0.12}
```

Use scale factors to exaggerate specific axes. A common use case is `z_scale` to amplify altitude:

```python
# Exaggerate altitude by 10x for better visibility
aspect = compute_aspect_ratio(
    span, supports, insulators,
    x_scale=1.0, y_scale=1.0, z_scale=10.0
)
# e.g. {'x': 1.0, 'y': 0.45, 'z': 1.2}
```

!!! note
    The function normalizes each axis range by the maximum range across all three axes, then multiplies by the corresponding scale factor. With default scales ($1.0$), the largest axis always has ratio $1.0$.

### Injecting aspect ratio into preview_line3d

Pass the computed aspect ratio directly to `preview_line3d`:

```python
fig = go.Figure()
plt_engine.preview_line3d(fig, aspect_ratio=aspect)
fig.show()
```

When `aspect_ratio` is provided, the layout is set to `aspectmode="manual"` with the given values, regardless of the `view` argument.

### Using set_layout directly

The `set_layout` function controls the 3D scene layout (axis labels, aspect ratio, camera). It is called internally by `preview_line3d`, but you can also use it directly for finer control:

```python
from mechaphlowers.plotting.plot import set_layout

fig = go.Figure()
# ... add traces manually ...

# Option 1: automatic layout (aspectmode="data")
set_layout(fig, auto=True)

# Option 2: compact layout (aspectmode="manual", default ratio)
set_layout(fig, auto=False)

# Option 3: custom aspect ratio
set_layout(fig, aspect_ratio={'x': 1.0, 'y': 0.5, 'z': 5.0})
```

!!! important
    When `aspect_ratio` is provided, the `auto` parameter is ignored for the aspect ratio. The layout always uses `aspectmode="manual"` with the provided values.

## Complete example

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import mechaphlowers as mph
from mechaphlowers.data.catalog.catalog import sample_cable_catalog
from mechaphlowers.plotting.plot import PlotEngine
from mechaphlowers.plotting import compute_aspect_ratio

# 1. Define section data
section_array = mph.SectionArray(
    pd.DataFrame({
        "name": ["1", "2", "3", "4"],
        "suspension": [False, True, True, False],
        "conductor_attachment_altitude": [30, 30, 20, 20],
        "crossarm_length": [10, 1.5, 5, 10],
        "line_angle": [34, 34, 30, 0],
        "insulator_length": [3.78, 2.17, 3, 1],
        "span_length": [100, 350, 200, 0],
        "insulator_mass": [100, 350, 200, 0],
        "load_mass": [0, 0, 0, np.nan],
        "load_position": [0, 0, 0, np.nan],
    })
)

# 2. Create balance engine and solve
cable_array = sample_cable_catalog.get_as_object(["ASTER600"])
balance = mph.BalanceEngine(section_array=section_array, cable_array=cable_array)

# 3. Build plot engine
pe = PlotEngine.builder_from_balance_engine(balance_engine=balance)

# 4. Plot 3D with custom aspect ratio
fig = go.Figure()
pe.preview_line3d(fig, aspect_ratio=aspect)
fig.show()

# 5. Plot 2D profile view
fig_2d = go.Figure()
pe.preview_line2d(fig_2d, view="profile", frame_index=0)
fig_2d.show()

# 6. Compute data-driven aspect ratio with altitude exaggeration
span, supports, insulators = pe.get_points_for_plot()
aspect = compute_aspect_ratio(span, supports, insulators, z_scale=5.0)
set_layout(fig, aspect_ratio=aspect)
fig.show()
```
