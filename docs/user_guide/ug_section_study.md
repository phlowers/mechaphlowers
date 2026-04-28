# Section Study

## Overview

`SectionStudy` is the main user-facing facade that bundles all engines needed to study a power-line section.
It wraps `BalanceEngine`, `PositionEngine`, `PlotEngine`, `ThermalEngine`, and `Guying` into a single object,
providing a simplified workflow with additional safety features:

* **Automatic rollback** on solver errors — the engine state is restored to the snapshot taken before the solve attempt.
* **Intermediate warm-start** — before solving extreme weather conditions, a preliminary solve at default conditions ($T = 15\,°C$, wind = 0, ice = 0) improves convergence.
* **State save / restore** — take snapshots and roll back to any previous state using the Memento pattern.

## Quick start

```python
from mechaphlowers import SectionStudy, CableArray, SectionArray

study = SectionStudy(cable_array, section_array)

# 1. Adjustment solve (computes L_ref)
study.solve_adjustment()

# 2. Change-of-state solve
study.solve_change_state(wind_pressure=200, new_temperature=90)

# 3. Retrieve results
points = study.get_supports_points()
data = study.get_data_spans()
```

## Creating a SectionStudy

A `SectionStudy` requires the same inputs as a `BalanceEngine`: a `CableArray` and a `SectionArray`.

```python
import pandas as pd
from mechaphlowers import SectionStudy, CableArray, SectionArray

section_array = SectionArray(
    pd.DataFrame({
        "name": ["1", "2", "3", "4"],
        "suspension": [False, True, True, False],
        "conductor_attachment_altitude": [50, 100, 50, 50],
        "crossarm_length": [10, 10, 10, 10],
        "line_angle": [0, 0, 0, 0],
        "insulator_length": [3, 3, 3, 3],
        "span_length": [500, 500, 500, float("nan")],
        "insulator_mass": [100, 50, 50, 100],
        "load_mass": [0, 0, 0, 0],
        "load_position": [0, 0, 0, 0],
    }),
    sagging_parameter=2000,
    sagging_temperature=15,
)
section_array.add_units({"line_angle": "grad"})

cable_array = CableArray(...)  # your cable data

study = SectionStudy(cable_array, section_array)
```

Custom span and deformation models can be passed at construction:

```python
from mechaphlowers.core.models.cable.span import CatenarySpan
from mechaphlowers.core.models.cable.deformation import DeformationRte

study = SectionStudy(
    cable_array,
    section_array,
    span_model_type=CatenarySpan,
    deformation_model_type=DeformationRte,
)
```

## Solving

### Adjustment

The adjustment solve computes the reference length $L_{ref}$ from sagging conditions (no wind, no ice, sagging temperature).

```python
study.solve_adjustment()
```

### Change of state

The change-of-state solve computes insulator chain positions for given weather conditions.

```python
study.solve_change_state(
    wind_pressure=200,       # Pa
    ice_thickness=0.01,      # m
    new_temperature=-10,     # °C
    wind_sense="anticlockwise",
)
```

!!! important
    `solve_adjustment()` must be called before `solve_change_state()`. If it has not been called, `SectionStudy` will trigger it automatically with a warning.

### Intermediate warm-start

When the requested conditions differ from the defaults ($T = 15\,°C$, wind = 0, ice = 0), `SectionStudy` automatically performs an intermediate solve at the default conditions first.
This provides a better starting point for the solver, improving convergence for extreme conditions.

The intermediate result is accessible via the `intermediate_memento` property:

```python
study.solve_change_state(wind_pressure=500, new_temperature=-20)

# Inspect the intermediate state (T=15°C, wind=0, ice=0)
intermediate = study.intermediate_memento
print(intermediate.nodes_dxdydz)
```

!!! note
    When the requested conditions match the defaults, the intermediate step is skipped and `intermediate_memento` is `None`.

### Automatic rollback

If the solver fails during `solve_adjustment()` or `solve_change_state()`, the engine state is automatically restored to the snapshot taken before the solve was attempted. A `SolverError` is still raised so you can handle it:

```python
from mechaphlowers.entities.errors import SolverError

try:
    study.solve_change_state(wind_pressure=99999, new_temperature=-100)
except SolverError:
    # Engine state is unchanged — safe to retry with different parameters
    study.solve_change_state(wind_pressure=200, new_temperature=-10)
```

## State management

`SectionStudy` exposes manual save / restore methods using the Memento pattern.

### Save and restore

```python
# Save state after adjustment
study.solve_adjustment()
memento = study.save_state()

# Solve with one set of conditions
study.solve_change_state(wind_pressure=200, new_temperature=90)
result_1 = study.get_supports_points().copy()

# Restore to post-adjustment state
study.restore_state(memento)

# Solve with different conditions — starts from the same base state
study.solve_change_state(wind_pressure=0, new_temperature=-20)
result_2 = study.get_supports_points()
```

!!! important
    `restore_state()` automatically notifies downstream engines (`PositionEngine`, `PlotEngine`) so their data stays consistent.

### Using the Caretaker directly

For advanced use cases, you can also use the `BalanceEngineCaretaker` directly with any `BalanceEngine`:

```python
from mechaphlowers import BalanceEngine, BalanceEngineCaretaker

engine = BalanceEngine(cable_array, section_array)
caretaker = BalanceEngineCaretaker(engine)

engine.solve_adjustment()
memento = caretaker.save()

engine.solve_change_state(new_temperature=90)
caretaker.restore(memento)  # engine is back to post-adjustment state
```

!!! warning
    When using `BalanceEngineCaretaker` directly, observer notification is not automatic. Call `engine.notify()` manually if a `PositionEngine` or `PlotEngine` is attached.

## Accessing sub-engines

`SectionStudy` provides access to all sub-engines through properties:

| Property | Type | Creation |
|---|---|---|
| `balance_engine` | `BalanceEngine` | Eager (at construction) |
| `position_engine` | `PositionEngine` | Eager (at construction) |
| `plot_engine` | `PlotEngine` | Lazy (on first access) |
| `thermal_engine` | `ThermalEngine` | Lazy (on first access) |
| `guying` | `Guying` | Lazy (on first access) |

```python
# Direct access to sub-engines when needed
balance = study.balance_engine
position = study.position_engine

# Lazy engines are created on first access
plot = study.plot_engine  # imports plotly only now
```

## Retrieving results

### Support points

```python
points = study.get_supports_points()  # shape: (n_supports, 3)
```

### Span points

```python
points = study.get_spans_points(frame="section")
```

### Span data

```python
data = study.get_data_spans()
# Returns a dict with keys: span_length, elevation, parameter,
# tension_sup, tension_inf, L0, horizontal_distance, arc_length, T_h, sag, sag_s2
```

## Adding loads

```python
import numpy as np

study.add_loads(
    load_position_distance=np.array([150, 200, 0, np.nan]),
    load_mass=np.array([500, 70, 0, np.nan]),
)
```

## Plotting

Since `PlotEngine` is lazily created, you can use it directly from `SectionStudy`:

```python
import plotly.graph_objects as go

study.solve_adjustment()
study.solve_change_state(wind_pressure=200, new_temperature=90)

fig = go.Figure()
study.plot_engine.preview_line3d(fig)
fig.show()
```
