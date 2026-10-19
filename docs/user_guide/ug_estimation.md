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
| `estimate_temperature` | Cable temperature (°C)        | [-50, 200]       | 3D distance                            |
| `estimate_wind`        | Wind pressure (Pa)            | [-1000, 1000]    | 3D distance                            |
| `estimate_load`        | Load mass (kg)                | [0, 2000]        | 3D distance                            |
| `estimate_rope`        | Rope length (m)               | [0.1, 10]        | Vertical distance above the obstacle   |

The other climatic conditions (temperature, wind pressure, ice thickness) can be fixed through optional arguments. A generic `estimate()` method also accepts any user-defined objective function.

!!! Warning

    `estimate_load` and `estimate_rope` cannot be used on a span split by a virtual support (`EstimateOnSplitSupportError`).
    For `estimate_load`, an existing load on the target span is ignored and replaced by the estimated one.

## Optimization methods

The root-finding algorithm is chosen with the `method` argument of `EstimationEngine`.

| Method           | Description                                                                                          |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| `BrentMethod`    | Default. Brent's method (`brentq`). Falls back to bisection if the interval does not bracket a root. |
| `BisectionMethod` | Simple bisection.                                                                                    |

The estimation module uses Brent's method by default because it is robust (it keeps the root bracketed, so it always converges on a valid interval) and fast (superlinear convergence, close to quadratic near the root).

Both take a tolerance `tol` and a maximum number of iterations `maxiter`, and require the objective to change sign over `bounds`. Custom algorithms can be used by implementing the `OptimizationMethod` protocol (a `solve(objective, bounds)` method).

## Result

Every estimation returns an `EstimationResult` with:

- `value`: the estimated value
- `residual`: final residual $|f(value)|$
- `iterations`: number of iterations
- `converged`: whether the algorithm converged within tolerance

!!! Warning

    Always check `converged`. If the objective does not change sign over `bounds`, no root can be bracketed and the best bound is returned.

## Example

```python
import numpy as np

from mechaphlowers.core.models.estimation import BrentMethod, EstimationEngine

engine = EstimationEngine(study, method=BrentMethod(tol=0.01))

result = engine.estimate_temperature(
    span_index=0,
    obstacle_point=np.array([150.0, 0.0, 5.0]),
    target_distance=8.0,
    bounds=(0.0, 200.0),
)
print(result.value, result.converged)
```
