# Deformation

The deformation class implements the cable deformation models described in [Cable modeling](ug_cable_model.md#physics-based-cable-modeling).

This class is called in two situations:

- When computing $L_{ref}$, which is done only one time
- When solving the new state after giving new wind pressure, ice thickness and temperature in `find_parameter_solver.py`. In this case, the solver calls `IDeformation` many times in order to find the solution

In the former case, the computation is made without wind and ice, and the attribute `current_temperature` is equal to `sagging_temperature` (usually 15°C). That is why `current_temperature` equals `sagging_temperature` by default when the model is created.

In the latter case, the wind and ice are taken into account through the tension, and the new temperature is given by assigning `current_temperature`. The assigned temperature is used by the stress-strain law at the next evaluation: the law is prepared again only when the temperature (or the state) changes, so assigning the same values at each solver iteration has no extra cost. Assign a new array rather than modifying the current one in place, because in-place modifications are not detected.

The engine does this automatically in `solve_change_state(new_temperature=...)`. Models are normally built with `deformation_model_builder`:

```python
from mechaphlowers.core.models.cable.deformation import deformation_model_builder

deformation_model = deformation_model_builder(
    cable_array,
    span_model,
    sagging_temperature=np.full(span_model.parameter.shape, 15.0),
)
deformation_model.current_temperature
# array of 15.0

# input new temperature after a state change
deformation_model.current_temperature = np.full(span_model.parameter.shape, 25.0)
```

## Stress-strain laws

The constitutive behaviour of the cable (the relation between mean tension, temperature and strain) is described by a **stress-strain law**. Laws are in the module [`deformation_laws`][mechaphlowers.core.models.cable.deformation_laws]. Two interfaces are defined:

- `IStressStrainLaw`: the law itself. It is built once from the cable data (SI units) and is immutable. Its `prepare(temperature, state)` method returns a prepared law
- `IPreparedLaw`: the law with the temperature and the loading history frozen. It evaluates the strains used by the solvers: `epsilon` (total strain relative to the unstressed length at the reference temperature), `epsilon_0` (total strain relative to 0°C), and the thermal parts `epsilon_therm`, `epsilon_therm_0`

The deformation model caches the prepared law and prepares it again only when `current_temperature` or the loading history `state` is assigned. The solvers, which call the deformation model many times with the same temperature and history, therefore pay the preparation cost once.

All strains are **total** strains (mechanical plus thermal). For multi-material cables, mechanical and thermal strains cannot be separated, because each material has its own thermal strain. The solver equation only uses the total strain:

$$
\frac{L - L_0}{L_0} = \varepsilon_0
$$

where $L$ is the cable length of the span, and $L_0$ the unstressed length at 0°C.

### Linear law

The historical RTE model is the `LinearLaw`, with a linear elastic behaviour:

$$
\varepsilon = \frac{T_{mean}}{E S} + \alpha (\theta - \theta_{ref})
$$

with $E$ the Young modulus, $S$ the section, $\alpha$ the dilatation coefficient, $\theta$ the temperature and $\theta_{ref}$ the temperature of reference. This law has no loading history.

### Polynomial law

The `PolynomialLaw` models a cable made of one or two materials working in parallel, such as an ACSR cable with an aluminium **conductor** and a steel **heart**. It has a non-linear, permanent-strain behaviour, which depends on the loading history.

**Stresses.** Each material $i$ (conductor $c$ with coefficients `a0` to `a4`, heart $h$ with coefficients `b0` to `b4`) has a strain $x_i$ that depends on the total strain $\varepsilon$ and on its own thermal strain:

$$
x_i = \varepsilon - \alpha_i (\theta - \theta_{ref})
$$

The cable stress is the sum of the reduced stresses of the materials, where the reduced stress of a material is its stress multiplied by its section divided by the total section:

$$
\sigma = \frac{T_{mean}}{S} = \sum_i \tilde\sigma_i(x_i)
$$

The catalog coefficients are already reduced: no section ratio is applied in the code. Thus the stresses of the materials can be summed directly.

**Behaviour of each material.** Let $P_i(x) = \sum_{k=0}^{4} c_{i,k} x^k$ be the virgin (first loading) curve of material $i$, with $c_{c,k}$ = `a`$k$ and $c_{h,k}$ = `b`$k$. Let $x_{max,i}$ be the highest mechanical strain ever reached by material $i$ (the loading history, zero for a virgin cable), and $E_i$ its unloading modulus. The permanent strain of the material is:

$$
x_{p,i} = x_{max,i} - \frac{P_i(x_{max,i})}{E_i}
$$

| zone | condition | reduced stress $\tilde\sigma_i$ |
|---|---|---|
| first loading (virgin curve) | $x_i \geq x_{max,i}$ | $P_i(x_i)$ |
| unloading and reloading (elastic line) | $x_{p,i} \leq x_i < x_{max,i}$ | $E_i \, (x_i - x_{p,i})$ |
| slack (no compression) | $x_i < x_{p,i}$ | $0$ |

Beyond the first maximum of $P_i$, the virgin curve is replaced by a constant stress plateau equal to its maximum.

!!! note "Modelling choice"
    The plateau beyond the first maximum of the virgin polynomial is a modelling choice made to keep the stress non-decreasing. It has not been validated against reference data. When the total stress exceeds the stress of the plateau, a `ConvergenceError` is raised.

Consequences:

- $\sigma(\varepsilon)$ is continuous, non-decreasing and piecewise polynomial, with at most four breakpoints. Below the loading history, it is piecewise linear
- At high temperature, the conductor (larger dilatation coefficient) can become slack, and only the heart carries the load. This is the usual knee point of two-material cables
- A linear material (for example, `b2` to `b4` equal to zero) unloads on its own curve and does not take permanent strain
- The polynomial is only valid on a limited strain range (for example, about 0.5% for the `a*` coefficients of the NARCISSE cable): results outside this range are not physical

**Equivalent quantities.** For approximations (such as the initial guess of the solvers) and for the unstressed length, the law uses equivalent values:

$$
E_{eq} = \sum_i E_i \qquad \alpha_{eq} = \frac{\sum_i \alpha_i E_i}{\sum_i E_i}
$$

The unstressed length at the reference temperature $L_{\theta_{ref}}$ and the unstressed length at 0°C $L_0$ are linked by $L_{\theta_{ref}} = L_0 (1 + \alpha_{eq} \theta_{ref})$.

**Unloading modulus.** The modulus $E_i$ is the linear coefficient of the polynomial (`a1` for the conductor, `b1` for the heart). If this coefficient is zero, the modulus is read from `young_modulus_conductor` or `young_modulus_heart`. A material whose coefficients are all zero is ignored: a cable with only a conductor is a one-material law.

### Selecting the law

`deformation_law_builder` selects the law from the cable data:

- `PolynomialLaw` if any coefficient of degree 2 or more is non-zero (`a2` to `a4` or `b2` to `b4`)
- `LinearLaw` otherwise

The law can also be given explicitly to `deformation_model_builder(..., law=...)`.

### Catalog columns for two-material cables

The following optional columns of the cable data are used by the polynomial law (see [Input data](ug_input.md#cable)):

| column | unit | used when | default |
|---|---|---|---|
| `young_modulus_conductor` | MPa | `a1` is zero | none (required in that case) |
| `young_modulus_heart` | MPa | `b1` is zero | none (required in that case) |
| `dilatation_coefficient_conductor` | $°C^{-1}$ | always | `dilatation_coefficient` |
| `dilatation_coefficient_heart` | $°C^{-1}$ | always | `dilatation_coefficient` |

### Numerical strategy

The polynomial law is solved with numpy only, without any root-finding library:

- if all the materials stay below their loading history, $\sigma(\varepsilon)$ is linear on each part and the strain is given in closed form
- otherwise, the strain is found by Newton iterations restricted to the polynomial segment that contains the solution, with a bisection fallback. The last solution of each span is used as initial guess, because solvers call the law with close tensions

## Loading history

The loading history of a cable is stored in a `DeformationState`. It is the highest mechanical strain reached by each material, for each span, measured at the reference temperature (so it does not depend on the temperature). A `DeformationState` is immutable: each operation returns a new state.

For linear cables, the state is empty and the history has no effect.

### BalanceEngine API

The history is handled by the [balance engine](ug_balance_engine.md#loading-history). A typical study first solves the worst past case, commits it, then solves the other cases with the permanent elongation:

```python
engine = BalanceEngine(cable_array=cable_array, section_array=section_array)
engine.solve_adjustment()

# worst past case
engine.solve_change_state(ice_thickness=0.02, new_temperature=-5)
engine.commit_deformation_state()

# other cases, with the history
engine.solve_change_state(new_temperature=40)
engine.solve_change_state(wind_pressure=500, new_temperature=15)

# forget the history: the cable is virgin again
engine.reset_deformation_state()
```

- `engine.commit_deformation_state()`: raises the history where the last solved situation exceeds it. It does not change the solution of the last case
- `engine.ratchet = True` (or `BalanceEngine(..., ratchet=True)`): the history is committed automatically after each `solve_change_state`. This makes the result depend on the sequence of cases. With `ratchet = False` (default), the history is frozen, and each case is computed independently from the committed history
- `engine.reset_deformation_state()`: the history is cleared
- `engine.deformation_state`: the current `DeformationState`. It can also be set to a given state

For a span with a load, the history is the maximum of the two semi-spans, because the most stressed part of the span sets the permanent elongation.

The history is saved by the memento of the engine (`BalanceEngineMemento`), so a restored state also restores its history. The history is kept by `engine.reset()`, as long as the number of spans is unchanged. Use `reset_deformation_state()` to clear it.