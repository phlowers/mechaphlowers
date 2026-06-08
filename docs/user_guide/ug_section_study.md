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

If the solver fails during `solve_change_state()`, the engine state is automatically restored to the snapshot taken before the solve was attempted. A `SolverError` is still raised so you can handle it:

```python
from mechaphlowers.entities.errors import SolverError

try:
    study.solve_change_state(wind_pressure=99999, new_temperature=-100)
except SolverError:
    # Engine state is unchanged — safe to retry with different parameters
    study.solve_change_state(wind_pressure=200, new_temperature=-10)
```

For `solve_adjustment()` **without** manipulations, the same rollback applies. When manipulations are active, `solve_adjustment()` first runs on a temporary clean engine (leaving the current engine untouched), so the state is naturally safe if the solver fails.

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

## Manipulations

`SectionStudy` provides methods to alter the geometry and insulator properties of supports as overlays, without modifying the original `SectionArray`. Manipulations are registered on the `study` object and are applied when `solve_adjustment()` is called: a clean adjustment is first solved on the original geometry, then a new engine is built from the manipulated copy and receives the injected $L_{ref}$.

!!! important
    Manipulations must be registered **before** calling `solve_adjustment()`. After `solve_adjustment()` with active manipulations, `plot_engine` and `guying` are reset and recreated lazily on next access.

### Support Manipulation

`modify_support` applies **additive offsets** to `conductor_attachment_altitude` and/or `crossarm_length` for specified supports. Internally the `Manipulation` object produces a new `SectionArray` with the offsets baked in; the original section array is never modified.

The input is a dictionary where keys are support indices (0-based) and values are dicts with optional keys `"y"` (crossarm length offset) and `"z"` (altitude offset), both in meters.

```python
# Raise support 1 by 2 m and shorten its crossarm by 1 m
study.modify_support({1: {"z": 2.0, "y": -1.0}})

# Modify several supports at once
study.modify_support({0: {"z": 0.5}, 2: {"y": 3.0}})

study.solve_adjustment()
study.solve_change_state(new_temperature=15.0)

# Restore original geometry
study.reset_support()
```

!!! note
    Manipulations are **additive**: calling `modify_support` multiple times stacks the offsets.
    `reset_support` clears all accumulated offsets and restores the original geometry.
    For each affected support, `counterweight_mass` is set to 0; unaffected supports keep their original value.

### Rope Manipulation

`add_rope` replaces the insulator length and mass for specified supports with rope values. Internally the `Manipulation` object produces a new `SectionArray` with the rope values baked in; the original section array is never modified.

The input is a dictionary where keys are support indices (0-based) and values are the rope length in meters. An optional `rope_lineic_mass` parameter (kg/m, default `0.01`) controls the mass per unit length.

```python
# Replace insulator properties for supports 1 and 2 with rope values
study.add_rope({1: 4.5, 2: 3.0})

# With a custom linear mass
study.add_rope({0: 2.0}, rope_lineic_mass=0.05)

study.solve_adjustment()
study.solve_change_state(new_temperature=15.0)

# Remove the rope overlay
study.reset_rope()
```

!!! note
    The rope overlay only affects `insulator_length` and `insulator_mass` (and the derived `insulator_weight`) for the listed supports.
    Unlisted supports keep their original insulator values.
    For each affected support, `counterweight_mass` is set to 0; unaffected supports keep their original value.
    The default linear mass can be changed globally via `options.data.rope_lineic_mass_default`.

### Virtual Support

`add_virtual_support` inserts intermediate supports into a line section. Internally the `Manipulation` object produces a new `SectionArray` with the virtual support rows inserted; the original section array is never modified. Each virtual support splits a given span at a specified horizontal distance from the left support. Because the number of supports changes, the full internal model is rebuilt while preserving observer bindings.

The input is a dictionary where keys are left-support indices (0-based, must not be the last support) and values are dicts with the following required keys:

| Key | Description |
|---|---|
| `"x"` | Distance from the left support (m) — must be strictly in `(-abs(crossarm_length[left_support]), abs(span_length) + abs(crossarm_length[right_support]))` |
| `"y"` | Lateral offset (m) — sets `line_angle = atan2(y, x)` on the left support |
| `"z"` | `conductor_attachment_altitude` of the virtual support (m) |
| `"insulator_length"` | Insulator length on the virtual support (m) |
| `"insulator_mass"` | Insulator mass on the virtual support (kg) |
| `"hanging_cable_point_from_left_support"` | Distance from the left support to the cable hanging point (m) — must be strictly in `(-abs(crossarm_length[left_support]), abs(span_length) + abs(crossarm_length[right_support]))`. Not used for computation currently. |

```python
# Insert a virtual support at 100 m into span 1
study.add_virtual_support({
    1: {"x": 100.0, "y": 0.0, "z": 55.0,
        "insulator_length": 3.0, "insulator_mass": 500.0,
        "hanging_cable_point_from_left_support": 100.0}
})

study.solve_adjustment()
study.solve_change_state(new_temperature=15.0)

# Remove all virtual supports
study.reset_virtual_support()
```

Multiple spans can be provided in one call, or via successive calls (overlays accumulate):

```python
study.add_virtual_support({
    0: {"x": 200.0, "y": 0.0, "z": 40.0, "insulator_length": 3.0, "insulator_mass": 500.0,
        "hanging_cable_point_from_left_support": 200.0},
    2: {"x": 200.0, "y": 10.0, "z": 62.0, "insulator_length": 3.0, "insulator_mass": 500.0,
        "hanging_cable_point_from_left_support": 200.0},
})
```

!!! note
    Virtual supports have `crossarm_length = 0` and `suspension = True`.  
    `counterweight_mass` is set to 0 for each virtual row.  
    All changes are reversible with `reset_virtual_support`; the original section array is never modified.

### Cable Shifting

`modify_cable` applies horizontal support shifting and span length modifications to the cable geometry. This is applied at solve time alongside the other manipulations.

- `shift_support`: array of horizontal offsets per support (m), length = number of supports; first and last values are forced to 0.
- `shorten_span`: array of span length reductions (m), length = number of spans; positive values shorten the span.

```python
import numpy as np

# Shift support 1 by 0.5 m, shorten the first span by 1 m
study.modify_cable(
    shift_support=np.array([0.0, 0.5, 0.0, 0.0]),
    shorten_span=np.array([1.0, 0.0, 0.0]),
)

study.solve_adjustment()
study.solve_change_state(new_temperature=15.0)

# Remove cable shifting
study.reset_cable()
```

!!! note
    Cable shifting modifies the effective $L_{ref}$ passed to the manipulated engine; it does not alter the geometry of supports.
