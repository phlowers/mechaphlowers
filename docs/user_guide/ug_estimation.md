# Estimation

## Overview

The estimation engine solves *inverse problems* on a solved `SectionStudy`: instead of computing the distance between the cable and an obstacle for a given condition, it finds the value of a physical variable that produces a **target distance**.

Typical questions:

- At which cable temperature does the cable reach 8 m from this obstacle?
- Which wind pressure brings the cable to a given distance from this obstacle?
- Which load mass, hung at a given position in the span, brings the cable to a given distance?
- Which rope length, set on a support, gives a target vertical clearance above this obstacle?

## Principle

For each estimated variable $x$, the engine builds an objective function:

$f(x) = d(x) - d_{target}$

where $d(x)$ is the distance between the cable and the obstacle point once the balance is solved with the variable set to $x$. The estimation consists in finding the root of $f$ over a search interval (`bounds`), using a root-finding method.

Each evaluation of $f$ runs a change of state in the balance engine. This is done on a copy of the study: the original `SectionStudy` is left untouched.

!!! Note

    The study must have been solved (`solve_adjustment`) before running an estimation.

## Available estimations

| Method                | Estimated variable            | Default bounds   | Distance used                          |
| --------------------- | ----------------------------- | ---------------- | -------------------------------------- |
| `estimate_temperature` | Cable temperature (°C)        | [-50, 200]       | Signed vertical distance (default)     |
| `estimate_wind`        | Wind pressure (Pa)            | [-1000, 1000]    | 3D distance                            |
| `estimate_load`        | Load mass (kg)                | [0, 2000]        | 3D distance                            |
| `estimate_rope`        | Rope length (m)               | [0.01, 10]       | Signed vertical distance (default)     |
| `estimate_load_relative` | Load mass (kg)              | [0, 2000]        | Cable altitude change at the load abscissa |

The `distance_type` argument of `estimate_temperature`, `estimate_wind`, `estimate_load` and `estimate_rope` selects the distance used as the target:

- `"distance_3d"`: unsigned 3D distance between the cable and the obstacle.
- `"horizontal"`: signed distance along the horizontal axis of the plane containing the obstacle and orthogonal to the span.
- `"vertical"`: signed vertical distance `z_cable - z_obstacle`. It is negative when the obstacle is above the cable.

`estimate_temperature` and `estimate_rope` default to `"vertical"`; `estimate_wind` and `estimate_load` default to `"distance_3d"`.

`estimate_load_relative` has no obstacle: it reads the cable altitude at `load_position_distance` on the current state of the study, then finds the load for which `new altitude - initial altitude` equals `target_distance` (negative when the cable goes down).

The other climatic conditions (temperature, wind pressure, ice thickness) can be fixed through optional arguments. A generic `estimate()` method also accepts any user-defined objective function.

For `estimate_wind`, set `wind_direction="clockwise"` (towards the user,
right) or `"anticlockwise"` (away from the user, left; the default), using
the same convention as `SectionStudy.solve_change_state`. Use non-negative
pressure bounds, such as `bounds=(0.0, 1000.0)`, to search only in the selected
direction. Negative pressures reverse that direction; the default signed
bounds remain unchanged.

!!! Warning

    `estimate_load`, `estimate_load_relative` and `estimate_rope` cannot be used on a span split by a virtual support (`EstimateOnSplitSupportError`).
    For `estimate_load` and `estimate_load_relative`, an existing load on the target span is ignored and replaced by the estimated one.

## Optimization methods

The root-finding algorithm is chosen with the `method` argument of `EstimationEngine`.

| Method           | Description                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| `BrentMethod`    | Default. Brent's method (`brentq`). Falls back to bisection if the interval does not bracket a root. |
| `BisectionMethod` | Simple bisection.                                                                                    |

The estimation module uses Brent's method by default because it is robust (it keeps the root bracketed, so it always converges on a valid interval) and fast (superlinear convergence, close to quadratic near the root).

Both take a tolerance `tol` and a maximum number of iterations `maxiter`, and require the objective to change sign over `bounds`. Custom algorithms can be used by implementing the `OptimizationMethod` protocol (a `solve(objective, bounds)` method).

The closest-point mode of `estimate_wind` (`target_distance=None`) minimizes the distance instead of finding a root. Its algorithm is chosen with the `minimizer` argument of `EstimationEngine`. It defaults to `BrentMinimizationMethod`, Brent's bounded minimization (`fminbound`): a derivative-free mix of golden-section search and parabolic interpolation. It takes the same `tol` and `maxiter` arguments. Custom algorithms can be used by implementing the `MinimizationMethod` protocol (a `minimize(objective, bounds)` method).

## Result

Every estimation returns an `EstimationResult` with:

- `value`: the estimated value
- `residual`: final residual $|f(value)|$
- `iterations`: number of iterations
- `converged`: whether the algorithm converged within tolerance

!!! Warning

    Always check `converged`. If the objective does not change sign over `bounds`, no root can be bracketed and the best bound is returned.

## Examples

The examples below share the same setup: a solved study of three spans, with an obstacle located under the middle span (span index 1), about 39 m below the cable at 15 °C.

```python
import numpy as np
import pandas as pd

from mechaphlowers import SectionArray, SectionStudy
from mechaphlowers.core.models.estimation import BrentMethod, EstimationEngine
from mechaphlowers.data.catalog import sample_cable_catalog

section_array = SectionArray(
    pd.DataFrame(
        {
            "name": ["1", "2", "3", "4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [50, 100, 50, 50],
            "crossarm_length": [10, 10, 10, 10],
            "line_angle": [0, 0, 0, 0],
            "insulator_length": [3, 3, 3, 3],
            "span_length": [500, 500, 500, float("nan")],
            "insulator_mass": [100, 50, 50, 100],
        }
    ),
    sagging_parameter=2000,
    sagging_temperature=15,
)
section_array.add_units({"line_angle": "grad"})
cable_array = sample_cable_catalog.get_as_object(["ASTER600"])

study = SectionStudy(cable_array, section_array)
study.solve_adjustment()

engine = EstimationEngine(study, method=BrentMethod(tol=0.01))
obstacle = np.array([750.0, 10.0, 20.0])  # x, y, z in meters
```

### Temperature

At which temperature is the cable at 36 m above the obstacle (signed vertical distance, the default)?

```python
result = engine.estimate_temperature(
    span_index=1,
    obstacle_point=obstacle,
    target_distance=36.0,
    bounds=(0.0, 150.0),
)
print(result.value, result.converged)  # ~74.7 °C, True
```

Other climatic conditions can be fixed, and another distance can be targeted:

```python
result = engine.estimate_temperature(
    span_index=1,
    obstacle_point=obstacle,
    target_distance=36.0,
    bounds=(0.0, 150.0),
    wind_pressure=200.0,
    distance_type="distance_3d",
)
```

### Wind

Which wind pressure brings the cable at 42 m (3D distance, the default) from the obstacle?

```python
result = engine.estimate_wind(
    span_index=1,
    obstacle_point=obstacle,
    target_distance=42.0,
    bounds=(0.0, 1000.0),
    wind_direction="clockwise",
    new_temperature=15.0,
)
print(result.value, result.converged)  # ~285 Pa, True
print(result.extra["distance"])  # distance at the solution, ~42 m
```

#### Closest point (`target_distance=None`)

Instead of a target distance, pass `target_distance=None` to find the wind pressure within `bounds` that brings the cable closest to the obstacle. The result also contains `extra["distance"]`, the distance at that wind pressure. It minimizes the absolute value of the distance with the engine `minimizer`. This mode assumes the distance has a single local minimum within `bounds`. If the minimum lies outside `bounds`, the closest bound is returned with `converged=False`.

```python
result = engine.estimate_wind(
    span_index=1,
    obstacle_point=obstacle,
    target_distance=None,
    bounds=(-1000.0, 1000.0),
    wind_direction="clockwise",
    new_temperature=15.0,
)
print(result.value, result.extra["distance"])  # wind at minimum distance, and that distance
```

### Load

Which mass, hung at 250 m from the left support of the span, brings the cable at 37 m (3D distance, the default) from the obstacle?

```python
result = engine.estimate_load(
    span_index=1,
    obstacle_point=obstacle,
    target_distance=37.0,
    load_position_distance=250.0,
    bounds=(0.0, 500.0),
)
print(result.value, result.converged)  # ~95 kg, True
```

### Load relative to the initial altitude

Which mass, hung at 250 m from the left support of the span, lowers the cable by 1 m at its own abscissa? No obstacle is needed, and the target is negative when the cable goes down.

```python
result = engine.estimate_load_relative(
    span_index=1,
    target_distance=-1.0,
    load_position_distance=250.0,
    bounds=(0.0, 500.0),
)
print(result.value, result.converged)  # ~40 kg, True
print(result.extra["x_shift"])  # initial x_cable - final x_cable, in meters
```

### Rope

Which rope length, set on support 2 (right support of span 1), gives a vertical clearance of 38.5 m above the obstacle?

```python
result = engine.estimate_rope(
    support_index=2,
    span_index=1,
    obstacle_point=obstacle,
    target_vertical_distance=38.5,
    bounds=(1.0, 10.0),
)
print(result.value, result.converged)  # ~5.7 m, True
```

The rope linear mass (`rope_lineic_mass`) and the climatic conditions can also be set.

### Custom objective

`estimate()` accepts any function of the variable whose root is searched:

```python
result = engine.estimate(lambda x: x**2 - 4, bounds=(0.0, 5.0))
print(result.value)  # ~2.0
```
