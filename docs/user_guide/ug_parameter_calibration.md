# Parameter calibration

In this page, we will explain the computation used by `param_calibration()` method in `src/mechaphlowers/data/measures.py`.


## Context

Before performing any mechanical computation, it is needed to input a sagging parameter at a sagging temperature (usually 15°C).
Usually, this parameter is not directly measured at the wanted sagging temperature, but at a random cable temperature, using the PAPOTO method for example.

When estimating the sagging parameter using measurement on the field, the usual procedure is the following:

- Estimate the parameter on the field.
- Compute the cable temperature during the measurement using the ThermOHL package.
- Use the `param_calibration()` method to compute the sagging parameter at the desired sagging temperature.

The `param_calibration()` method takes for input the parameter measured, and a cable temperature, and returns the parameter at at the desired sagging temperature.

## Equations

We have the following input values:

- $p_m$ = Measured parameter (m)
- $\theta_m$ = Cable temperature during the measure (°C)
- $\theta_s$ = sagging temperature : wanted cable temperature (°C)

Let $f(p)$ be the function that computes the parameter at temperature $\theta_m$ depending on the sagging parameter $p$:

The goal is to find the sagging parameter $p_s$ where $f(p_s) = p_m$.
This is equivalent to finding the root of the following function:

$\delta(p) = f(p) - p_m$

In order to find this root, we use a Newton-Raphson method. However, since the required precision for this computation is not very high and the initial guess $p_0$ is already close to the true value, performing only one iteration of the Newton-Raphson method is sufficient.

The Newton-Raphson update formula is:

$p_1 = p_0 - \frac{\delta(p_0)}{\delta'(p_0)}$

where $\delta'(p)$ is the derivative of $\delta(p)$, approximated using finite differences.

## Initial parameter guess

The initial guess $p_0$ is computed by setting a state where:

- sagging_temperature is $\theta_m$ (and not $\theta_s$ )
- sagging_parameter is $p_m$
- changing state to $\theta_s$  and reading the sagging parameter at this state.


When computing the function $f(p)$, we compute the following states:

| Initial state | Change state |
| ------------- | ------------ |
| $\theta_s$    | $\theta_m$   |
| $p_s$         | $p_m$        |


To compute the initial guess, we reverse the state change process described above:

| Initial state | Change state |
| ------------- | ------------ |
| $\theta_m$    | $\theta_s$   |
| $p_m$         | $p_0$        |

And we assume that $p_0$ is a good enough approximation of the parameter $p_s$ at $\theta_s$.

## Uncertainty estimation (Monte Carlo)

After estimating the PAPOTO parameter with `PapotoParameterMeasure`, you can quantify the sensitivity of the result to angle measurement errors using the `uncertainty()` method.

### Principle

The method applies a **Monte Carlo** approach:

1. For each of the 10 angle inputs (`HL`, `VL`, `HR`, `VR`, `H1`, `V1`, `H2`, `V2`, `H3`, `V3`), a random perturbation drawn uniformly from `[-angle_error, +angle_error]` is added to the original measurement.
2. The full PAPOTO computation is re-run for all draws simultaneously using NumPy vectorization.
3. Draws where the validity criterion is not satisfied are filtered out.
4. Statistics are returned for both the valid and non-valid populations.

### Usage

```python
from mechaphlowers.data.measures import PapotoParameterMeasure

papoto = PapotoParameterMeasure()

# First, estimate the parameter from field measurements
papoto(
    a=498.57,
    HL=0.0, VL=97.43, HR=162.61, VR=88.69,
    H1=5.11, V1=98.45, H2=19.63, V2=97.63,
    H3=97.15, V3=87.93,
)

# Then estimate the uncertainty using 1000 Monte Carlo draws
# with an angle error of ±0.01 grad
result = papoto.uncertainty(draw_number=1000, angle_error=0.01)

print(result["mean_parameter_valid_values"])   # mean parameter over valid draws
print(result["std_parameter_valid_values"])    # standard deviation
print(result["parameter_by_span_length"])      # mean / span length
print(result["number_non_valid_values"])       # draws that failed validity
```

### Output keys

| Key | Description |
|-----|-------------|
| `mean_parameter_valid_values` | Mean parameter over valid draws |
| `std_parameter_valid_values` | Standard deviation over valid draws |
| `min_parameter_valid_values` | Minimum parameter over valid draws |
| `max_parameter_valid_values` | Maximum parameter over valid draws |
| `parameter_by_span_length` | `mean_parameter_valid_values / a` |
| `number_non_valid_values` | Number of draws that failed validity |
| `mean_non_valid_values` | Mean parameter over non-valid draws (`NaN` if none) |
| `std_non_valid_values` | Std deviation over non-valid draws (`NaN` if none) |
| `min_all_values` | Minimum over all draws |
| `max_all_values` | Maximum over all draws |

!!! note
    `measure_method()` (or calling the object directly) must be invoked before `uncertainty()`. A `RuntimeError` is raised otherwise.