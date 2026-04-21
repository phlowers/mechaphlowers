# Input data

## Line Section

This paragraph describes the input data and the associated format needed to perform calculations on a line section in mechaphlowers.

!!! Units

    Unless specified otherwise, all units will be those of the International System of Units (SI).
    A list of these units can be found [here](https://en.wikipedia.org/wiki/International_System_of_Units).

In mechaphlowers a line section is described by the following data:

- a sagging parameter, denoted later as $p$,
- a sagging temperature (in Celsius degrees)
- for each support:
    - the name of the support
    - a boolean named `suspension` describing whether it's a suspension or tension support
    - the conductor attachment altitude
    - the crossarm length
    - the line angle (in degrees)
    - the insulator length
    - the insulator mass
    - an optional field ground altitude
    - an optional counterweight
- for each span:
    - the span length, denoted later as $a$.


!!! important

    Sagging parameter and temperature are assumed to be **the same for each span** in a line section - which doesn't necessarily reflect reality. This is not the case from the point you use physical engine to balance the line section.

!!! Warning

    Ground altitude is optional because it is autofilled if not provided.  
    Autofill rule: **ground_altitude = conductor_attachment_altitude - options_paramater**.  
    options_parameter is globally defined in options.ground.default_support_length and can be modified by user.


!!! Angle orientation convention

    The angles input in the section 


Input data should be organized in a table (for example a pandas dataframe), where each row describes one support with its following span, except the last row which only describes the last support (since it doesn't have a "following" span). Hence the last span length is expected to be "not a number", typically `numpy.nan`.

For example, a line section could be described by the following table:

|name|suspension|conductor_attachment_altitude|crossarm_length|line_angle|insulator_length|span_length|
|---------------|------|----|---|--|--|------|
|first support  |False |1     |12   |0  |0 |500   |
|second support |True  |-0.75 |10   |11 |4 |800   |
|third support  |False |0.5   |11.5 |0  |0 |      |

!!! Altitude

    Since the conductor attachment altitude is measured from the sea level, it may be negative.

In this example the span length between the first and second supports is 500 m, and the span length between the second and third support is 800 m. The line angle in the middle of the section is of 10 degrees.

You may use the following code to define this data and load it so that it can be used by mechaphlowers:

    import pandas as pd
    import numpy as np

    from mechaphlowers.entities.arrays import SectionArray


    input_df = pd.DataFrame({
        "name": ["first support", "second support", "third support"],
        "suspension": [False, True, False],
        "conductor_attachment_altitude": [1, -0.75, 0.5],
        "crossarm_length": [12, 10, 11.5],
        "line_angle": [0, 11, 0],
        "insulator_length": [0, 4, 0],
        "span_length": [500, 800, np.nan],
        "counterweight_mass": [0, 0, 0],
    })
    section_array = SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)
    print(section_array)

### Sagging default values

Sagging parameters and temperature have default values. In this way, user can vizualise section in the same time it is created.

**Rules**:

- sagging_temperature = 15°C
- sagging_parameter = equivalent_span $\times$ 5
- equivalent_span is the following, with $a_i$ the span length of the ith span:

$$
    L_{eq} = \sqrt{\frac{ \sum_{i \in span} a_i^3}{\sum_{i \in span} a_i}} 
$$

### Bundle number

When creating a SectionArray, you may add a `bundle_number` argument. `bundle_number` is set by default to 1.


    section_array = SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15, bundle_number=2)

### Support Manipulation

After creating a `SectionArray`, you can modify the geometry of individual supports using `support_manipulation`. This applies **additive offsets** to `conductor_attachment_altitude` and/or `crossarm_length`.

The input is a dictionary where keys are support indices (0-based) and values are dicts with optional keys `"y"` (crossarm length offset) and `"z"` (altitude offset), both in meters.

```python
# Raise support 1 by 2 m and shorten its crossarm by 1 m
section_array.support_manipulation({1: {"z": 2.0, "y": -1.0}})

# Modify several supports at once
section_array.support_manipulation({0: {"z": 0.5}, 2: {"y": 3.0}})
```

To restore the original geometry:

```python
section_array.reset_manipulation()
```

When using a `BalanceEngine`, the same methods are available and will automatically rebuild internal models while preserving observer bindings (e.g. `PlotEngine`):

```python
engine.support_manipulation({1: {"z": 2.0}})
engine.solve_adjustment()
engine.solve_change_state(new_temperature=15.0)

# Restore original geometry
engine.reset_manipulation()
```

!!! note

    Manipulations are **additive**: calling `support_manipulation` multiple times stacks the offsets.
    `reset_manipulation` always restores the values from before the first manipulation.
    For each affected support, `counterweight` is set to 0 in `.data`; unaffected supports keep their original counterweight.

### Rope Manipulation

`rope_manipulation` replaces the insulator length and mass for specified supports with rope values, **without modifying the underlying data**. The override is only visible through `.data`; `_data` remains unchanged.

The input is a dictionary where keys are support indices (0-based) and values are the rope length in meters. An optional `rope_lineic_mass` parameter (kg/m, default `0.01`) controls the mass per unit length.

```python
# Replace insulator properties for supports 1 and 2 with rope values
section_array.rope_manipulation({1: 4.5, 2: 3.0})

# With a custom linear mass
section_array.rope_manipulation({0: 2.0}, rope_lineic_mass=0.05)
```

To remove the rope overlay:

```python
section_array.reset_rope_manipulation()
```

The same API is available on `BalanceEngine`:

```python
engine.rope_manipulation({1: 4.5})
engine.solve_adjustment()
engine.solve_change_state(new_temperature=15.0)

engine.reset_rope_manipulation()
```

!!! note

    The rope overlay only affects `insulator_length` and `insulator_mass` (and the derived `insulator_weight`) for the listed supports.
    Unlisted supports keep their original insulator values.
    For each affected support, `counterweight` is set to 0 in `.data`; unaffected supports keep their original counterweight.
    The default linear mass can be changed globally via `options.data.rope_lineic_mass_default`.

### Virtual Support

`add_virtual_support` inserts intermediate supports into a line section **as an overlay** on `.data`, without ever modifying the underlying `_data`. Each virtual support splits a given span at a specified horizontal distance from the left support.

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
# insert a virtual support at 100 m into span 1
section_array.add_virtual_support({
    1: {"x": 100.0, "y": 0.0, "z": 55.0,
        "insulator_length": 3.0, "insulator_mass": 500.0,
        "hanging_cable_point_from_left_support": 100.0}
})
```

Multiple spans can be provided in one call, or via successive calls (overlays accumulate):

```python
section_array.add_virtual_support({
    0: {"x": 200.0, "y": 0.0, "z": 40.0, "insulator_length": 3.0, "insulator_mass": 500.0,
        "hanging_cable_point_from_left_support": 200.0},
    2: {"x": 200.0, "y": 10.0, "z": 62.0, "insulator_length": 3.0, "insulator_mass": 500.0,
        "hanging_cable_point_from_left_support": 200.0},
})
```

To remove all virtual supports:

```python
section_array.reset_virtual_support()
```

The same API is available on `BalanceEngine`. Because the number of supports changes, the full internal model is rebuilt while preserving observer bindings:

```python
engine.add_virtual_support({
    1: {"x": 100.0, "y": 0.0, "z": 55.0,
        "insulator_length": 3.0, "insulator_mass": 500.0,
        "hanging_cable_point_from_left_support": 100.0}
})
engine.solve_adjustment()
engine.solve_change_state(new_temperature=15.0)

engine.reset_virtual_support()
```

!!! note

    Virtual supports have `crossarm_length = 0` and `suspension = True`.  
    The `counterweight` column is set to 0 for each virtual row.  
    `elevation_difference` is recomputed from `data` (not `_data`) so it always reflects the inserted supports.  
    `_data` is never modified; all changes are overlay-only and reversible with `reset_virtual_support`.

## Cable

This paragraph describes the input data and the associated format needed about the cable properties in mechaphlowers.

A Cable is described using the following data:

- section in $mm^2$, denoted later as $S$
- diameter in $mm$, denoted later as $D$
- linear weight in $N/m$, denoted later as $\lambda$
- Young modulus in $GPa$, denoted later as $E$
- temperature of reference in $°C$
- dilatation coefficient in $°C^{-1}$, denoted later as $\alpha_{th}$
- coefficients of the polynomial model between stress and deformation in $GPa$


Similarly to line section data, input data should be organized in a table. However, the number of rows should be equal to 1: the attributes of a cable are the same on any span.

|section|diameter|linear_weight|young_modulus|dilatation_coefficient|temperature_reference|a0|a1|a2|a3|a4|b0|b1|b2|b3|b4|
|---|-----|--|--|--|-|-|--|-----|-------|-----------|-|-|-|-|-|
|450|30.5 |14|45|23|0|0|15|45000|2300000|-1800000000|0|0|0|0|0|

You may use the following code to define this data and load it so that it can be used by mechaphlowers:

```
import pandas as pd
import numpy as np

from mechaphlowers.entities.arrays import CableArray


input_df = pd.DataFrame({
	"section": [450],
	"diameter": [30.5],
	"linear_weight": [14],
	"young_modulus": [45],
	"dilatation_coefficient": [23],
	"temperature_reference": [0],
	"a0": [0],
	"a1": [15000],
	"a2": [45000000],
	"a3": [2300000000],
	"a4": [-1800000000000],
    "b0": [0],
    "b1": [0],
    "b2": [0],
    "b3": [0],
    "b4": [0],
})
cable_array = CableArray(input_df)
print(cable_array)
```
## External loads

External loads can be added to the cable array. We only support wind loads, but it's easy to extend this by adding more external loads. To add a weather load, you can use `BalanceEngine.solve_change_state(...)` method.

```python

# add weather
balance_engine.solve_change_state(
    ice_thickness=np.array([1e-2, 2.1e-2, 0.0, np.nan]),
    wind_pressure=np.array([240.12, 0.0, 12.0, np.nan]),
)

# get effects of the load: example with the load_angle
balance_engine_angles.cable_loads.load_angle
```


The following example shows how to add a wind load on the cable.

!!! Parameters unit

	The ice_thickness and wind_pressure are in meters and Pascal respectively.
	The format of those vectors is span oriented: their size is the same than the section but the last value is not used 
	That's why we put `np.nan` at the end.


!!! Wind direction convention

    Another attention point is that the wind load can be negative, which means that the wind is blowing in the opposite direction of the line. The sign convention is parameterized by wind_sense ("clockwise" or "anticlockwise").  
    If "clockwise": towards user (right), if "anticlockwise": away from user (left). Default to "anticlockwise".


Then you can display the effect of this load with:

- load_angle
- resulting_norm
- load_coefficient
- ice_load
- wind_load


## Support Shapes / position sets

Mechaphlowers provide a specific object to handle simple set representation: the SupportShape class.  
The idea is to provide a simple way to define a set of arms, each arm being represented by X,Y,Z coordinates of the arms.  
The arm length is computed from x, y coordinate.

Assumption for the representation:

- The support is centered with origin at the support ground.  
- Trunk is a vertical bar  
- base arms are on Y coordinate only, from the trunk  
- set point are on X coordinate only, from the edge of the base arms  

!!! Warning

    This is a simplified representation to visualize sets positions on the geometry.


```python
import plotly.graph_objects as go
import numpy as np  
import mechaphlowers as mph


fig = go.Figure()

pyl_shape = mph.SupportShape(
    name="pyl",
    xyz_arms=np.array(
        [
            [0, 0, 18.5],
            [0, 3, 16.5],
            [0, 6, 16.5],
            [0, 9, 16.5],
            [3, -3, 14.5],
            [-3, -3, 14.5],
            [0, -6, 14.5],
            [0, -9, 14.5],
        ]
    ),
    set_number=np.array([22, 28, 37, 44, 45, 46, 47, 55]),
)
mph.plotting.plot_support_shape(fig, pyl_shape)
fig.show()
```

--8<-- "docs/user_guide/assets/plot_support_shape.html"

