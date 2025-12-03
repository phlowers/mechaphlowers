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

$p_m$ = Measured parameter (m)
$\theta_m$ = Cable temperature during the measure (°C)
$\theta_s$ = sagging temperature : wanted cable temperature (°C)

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
| $p_s$           | $p_m$        |


To compute the initial guess, we reverse the state change process described above:

| Initial state | Change state |
| ------------- | ------------ |
| $\theta_m$    | $\theta_s$   |
| $p_m$         | $p_0$        |

And we assume that $p_0$ is a good enough approximation of the parameter $p_s$ at $\theta_s$.