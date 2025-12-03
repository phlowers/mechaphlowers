# Parameter estimation at 15°C

In this page, we will explain the computation used by `param_15_deg()` method in `src/mechaphlowers/data/measures.py`.


## Context

Before performing any mechanical computation, it is needed to input a sagging parameter at 15°C.
Usually, this parameter is not directly measured at 15°C, but at a random cable temperature, using the PAPOTO method for example.

When estimating the sagging parameter at 15°C using measurement on the field, the usual procedure is the following:
- Estimate the parameter on the field.
- Compute the cable temperature during the measurement using the ThermOHL package.
- Use the `param_15_deg()` method to compute the sagging parameter at 15°C.

The `param_15_deg()` method takes for input the parameter measured, and a cable temperature, and returns the parameter at 15°C.

## Equations

We have the following input values:

$p_m$ = Measured parameter (m)
$\theta_m$ = Cable temperature during the measure (°C)  

Let $f(p)$ be the function that computes the parameter at temperature $\theta_m$ depending on the parameter at 15°C $p$ (which is the sagging parameter):

The goal is to find the parameter p where $f(p) = p_m$.
This is equivalent to finding the root of the following function:

$\delta(p) = f(p) - p_m$

In order to find this root, we use a Newton-Raphson method. However, since the required precision for this computation is not very high and the initial guess $p_0$ is already close to the true value, performing only one iteration of the Newton-Raphson method is sufficient.

The Newton-Raphson update formula is given by:

$p_1 = p_0 - \frac{\delta(p_0)}{\delta'(p_0)}$

where $\delta'(p)$ is the derivative of $\delta(p)$, approximated using finite differences.

## Initial parameter guess

The initial guess $p_0$ is computed by setting a state where:
- sagging_temperature is $\theta_m$ (and not 15°C)
- sagging_parameter is $p_m$
- changing state to 15°C and reading the sagging parameter at this state.


When computing the function $f(p)$, we compute the following states:

| Initial state | Change state |
| ------------- | ------------ |
| 15°C          | $\theta_m$   |
| $p$           | $p_m$        |


To compute the initial guess, we reverse the state change process described above:

| Initial state | Change state |
| ------------- | ------------ |
| $\theta_m$    | 15°C         |
| $p_m$         | $p_0$        |

And we assume that $p_0$ is a good enough approximation of the parameter at 15°C $p$.