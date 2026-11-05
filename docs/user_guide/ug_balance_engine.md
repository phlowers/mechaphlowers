# Balance Engine

The `BalanceEngine` is the central API for computing insulator chain positions and cable tensions along a power line section. It handles two main scenarios:

- **Adjustment**: computing the reference cable length $L_0$ at sagging conditions (no weather, sagging temperature).
- **Change of state**: computing chain displacements and cable parameters under new weather conditions and/or temperature.

It also supports optional features such as adding point loads on spans and simulating cable shifting or shortening operations.

---

## Instantiation

A `BalanceEngine` requires a `CableArray` and a `SectionArray` as inputs. Both must be fully configured before the engine is created.

```python
import numpy as np
import pandas as pd
from mechaphlowers.data.catalog import sample_cable_catalog
from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.core.models.balance.engine import BalanceEngine

cable_array = sample_cable_catalog.get_as_object(["ASTER600"])

section_array = SectionArray(
    pd.DataFrame({
        "name": ["A", "B", "C", "D"],
        "suspension": [False, True, True, False],
        "conductor_attachment_altitude": [30.0, 50.0, 60.0, 65.0],
        "crossarm_length": [0.0, 0.0, 0.0, 0.0],
        "line_angle": [0.0, 0.0, 0.0, 0.0],
        "insulator_length": [3.0, 3.0, 3.0, 3.0],
        "span_length": [500.0, 300.0, 400.0, np.nan],
        "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
        "load_mass": [0.0, 0.0, 0.0, 0.0],
        "load_position": [0.0, 0.0, 0.0, 0.0],
    }),
    sagging_parameter=2000,
    sagging_temperature=15,
)
section_array.add_units({"line_angle": "grad"})

engine = BalanceEngine(cable_array=cable_array, section_array=section_array)
```

**Key attributes after construction:**

| Attribute | Description |
|---|---|
| `engine.support_number` | Number of supports (= `len(engine)`) |
| `engine.span_model` | The catenary span model |
| `engine.balance_model` | The balance model (insulator chains) |
| `engine.cable_loads` | Wind and ice load container |

---

## `solve_adjustment()`

Solves the insulator chain positions at sagging conditions. This establishes $L_0$ (the unstressed cable reference length) that is used as the reference for all subsequent change-of-state computations.

**Must be called before `solve_change_state()`** (or it will be triggered automatically with a warning).

```python
engine.solve_adjustment()

# Access the reference cable length (one value per span)
print(engine.L_ref)         # e.g. [500.8, 298.5, 401.7]
print(engine.parameter)     # sagging parameter per span
```

After this call, `engine.L_ref` is set. The sagging parameter and chain displacements are also updated inside the balance model.

---

## `solve_change_state()`

Solves the insulator chain positions under new weather conditions and/or a new temperature. All parameters are optional; omitting one uses its default value (see table below).

| Parameter | Default | Unit |
|---|---|---|
| `wind_pressure` | `0.0` | Pa |
| `ice_thickness` | `0.0` | m |
| `new_temperature` | `15.0` | °C |
| `wind_direction` | `"anticlockwise"` | — |

```python
engine.solve_adjustment()

engine.solve_change_state(
    wind_pressure=200.0,      # Pa
    ice_thickness=0.01,       # m
    new_temperature=0.0,      # °C
    wind_direction="anticlockwise",
)

# Results
print(engine.parameter)      # updated sagging parameter
print(engine.L_ref)          # reference length (unchanged by change of state but changed with manipulations)
```

**Wind sense convention:**

- `"anticlockwise"` (default): wind blows away from the viewer (left in the span plane).
- `"clockwise"`: wind blows towards the viewer (right in the span plane).

**Passing a scalar broadcasts to all spans; passing an array gives per-span control:**

```python
engine.solve_change_state(wind_pressure=np.array([200.0, 150.0, 100.0, 0.0]))
```

!!! warning
    If `solve_adjustment()` has not been called first, it is triggered automatically and a `BalanceEngineWarning` is emitted.

---

## Solver performance: banded Jacobian

Both adjustment and change-of-state computations use a Newton-Raphson solver.
At each iteration, the solver estimates the Jacobian by finite differences:
each column describes how the equilibrium equations change when one state
variable is perturbed. A dense finite-difference calculation requires one
model evaluation per state variable.

The balance equations have **local dependencies**: the moments at a node
depend only on that node and its immediate neighbours. With two state
variables and two equilibrium equations per node, non-zero Jacobian entries
lie within three positions of the diagonal. This is a **banded Jacobian** with
half-bandwidth $b = 3$:

$$
J_{ij} = 0 \quad \text{when } |i-j| > b.
$$

Columns separated by $2b+1$ positions affect disjoint sets of equations.
The solver can therefore perturb these columns together and recover their
derivatives from a single evaluation. This grouped finite-difference method
(Curtis-Powell-Reid) needs **7 model evaluations instead of one per state
variable** to build the balance-model Jacobian.

For point loads, the inner `LoadModel` solves each load's position independently
of the other loads, with the surrounding span geometry held fixed. Its
Jacobian consists of independent $2 \times 2$ blocks, giving $b = 1$ and
**3 evaluations instead of 2 per loaded span**. Reducing both outer and inner
evaluations is especially beneficial for sections with many loaded spans.

This optimisation is automatic; no change to the `BalanceEngine` API is needed.
The solver retains dense finite differences when the model does not declare
a bandwidth or when grouping would not reduce the number of evaluations.
It still stores a dense matrix and uses `numpy.linalg.solve`: this is an
optimisation of Jacobian construction, not a sparse linear solver.

### Measured performance

The following results compare `main` at `9a21da93` with the optimised branch
`refactor/optim-banded-jacobian-load-model` at `6041df16`, measured locally on
2026-11-05. Values are median times for a **fresh `solve_change_state()` call**
over five independently prepared engines; construction, adjustment and load
configuration are excluded.

| Scenario | Main | Optimised branch | Speedup |
|---|---:|---:|---:|
| 9 loaded spans | 5,308 ms | 545 ms | **9.7x** |
| 19 loaded spans | 21,090 ms | 539 ms | **39.1x** |
| 99 unloaded spans | 1,944 ms | 64 ms | **30.2x** |

The benchmarks used ASTER600 cable, 500 m spans, level 50 m supports, 3 m
insulators, and a sagging parameter of 2000 m at 15 °C. Change-of-state
conditions were -200 Pa wind pressure, 0.02 m ice thickness and 5 °C.
Loaded cases explicitly used `set_loads()` to place a 500 kg point load
200 m from the left support on every span.

Both revisions ran with the same dependencies (NumPy 2.2.5, SciPy 1.14.1,
pandas 2.3.1 and Pandera 0.33.1), Python 3.13.18 and single-threaded BLAS.

Across all nine benchmark scenarios, 108 paired numerical-output comparisons
passed for fresh and repeated solves. State and physical-output arrays used
relative tolerance $10^{-6}$ and absolute tolerance $10^{-5}$; equilibrium
objective arrays used absolute tolerance $10^{-2}$.

---

## `add_loads()`

Adds a point load (e.g. a maintenance device) on a span. Inputs are per-support arrays; the last support value must be `nan` (span-based convention).

```python
engine.add_loads(
    load_position_distance=np.array([150.0, 200.0, 0.0, np.nan]),  # m from left support
    load_mass=np.array([500.0, 70.0, 0.0, np.nan]),                # kg
)

engine.solve_adjustment()
engine.solve_change_state()
```

- `load_position_distance` must be in `[0, span_length]` for each span; a `ValueError` is raised otherwise.
- After the call, `engine.reset(full=False)` is triggered automatically to keep the engine consistent.

---

## `get_data_spans()`

Returns a dictionary summarising the key span-level results after a solve. All values are lists of length equal to the number of spans.

```python
engine.solve_adjustment()
engine.solve_change_state()

data = engine.get_data_spans()
```

| Key | Description | Unit |
|---|---|---|
| `span_length` | Horizontal span length | m |
| `elevation` | Elevation difference between supports | m |
| `parameter` | Sagging parameter | m |
| `tension_sup` | Tension at upper attachment | (configured output unit) |
| `tension_inf` | Tension at lower attachment | (configured output unit) |
| `L0` | Reference cable length | m |
| `horizontal_distance` | Projected horizontal distance | m |
| `arc_length` | Cable arc length | m |
| `T_h` | Horizontal component of tension | (configured output unit) |

---

## `reset()`

Re-initialises the engine. Use `full=True` to completely rebuild all internal models (span model, deformation model, balance model, solvers). Use `full=False` (default) for a lighter reset that keeps the span and deformation models but reloads loads and cable data.

```python
engine.reset(full=True)   # full rebuild — e.g. after a solver error
engine.reset(full=False)  # partial reset — e.g. after updating loads
```

!!! warning
    `reset(full=True)` rebuilds the internal models but does **not** clear `engine.L_ref`. If you need a fresh adjustment (for example after changing geometry, loads, or cable data), you must call `solve_adjustment()` again before `solve_change_state()`.

---

## Typical workflows

### Basic sag-tension

```python
engine = BalanceEngine(cable_array=cable_array, section_array=section_array)
engine.solve_adjustment()
engine.solve_change_state(wind_pressure=200.0, ice_thickness=0.01, new_temperature=-10.0)

data = engine.get_data_spans()
print(data["T_h"])
```

### Point load on a span

```python
engine = BalanceEngine(cable_array=cable_array, section_array=section_array)
engine.add_loads(
    load_position_distance=np.array([150.0, 0.0, 0.0, np.nan]),
    load_mass=np.array([500.0, 0.0, 0.0, np.nan]),
)
engine.solve_adjustment()
engine.solve_change_state()
```
