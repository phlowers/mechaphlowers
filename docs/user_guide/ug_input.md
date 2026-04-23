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
    - an optional field `ground_altitude`
    - an optional counterweight
    - an optional `support_height` (in meters, non-negative) — height of the support structure
    - an optional `x_offset` (in meters) — longitudinal offset of the attachment point
- for each span:
    - the span length, denoted later as $a$.


!!! important

    Sagging parameter and temperature are assumed to be **the same for each span** in a line section - which doesn't necessarily reflect reality. This is not the case from the point you use physical engine to balance the line section.

!!! Warning

    Ground altitude is optional because it is autofilled if not provided.  
    Autofill rule:

    $$
    \text{ground\_altitude} = \text{conductor\_attachment\_altitude} + \text{insulator\_length} - \text{support\_height} + \text{spacer\_height} - \text{foot\_to\_ground\_clearance}
    $$

    Where:

    - `support_height` is taken from the column if provided, otherwise from `options.ground.default_support_length` (default: 30 m).
    - `spacer_height` is the height contribution of the `Spacer` equipment (default: 0.2 m for bundle numbers 3 and 4, else 0 m).
    - `foot_to_ground_clearance` is defined in `options.ground.foot_to_ground_clearance` (default: 0.2 m). It is also called OO' distance.


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

## Equipment

Equipments are physical objects installed on the line that influence geometry calculations. They are passed to `SectionArray` at construction time.

### Spacer

A `Spacer` maintains separation between conductors in a bundle. It contributes to the `ground_altitude` computation when the bundle number is 3 or 4 (triangular or square bundles), adding its length (in meters) to the altitude offset.

For bundle numbers 1 and 2, the spacer contribution is zero.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `length`  | 0.2 m   | Physical length of the spacer |

```python
from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.entities.equipment import Spacer

# Default spacer (length=0.2 m)
spacer = Spacer()

# Custom spacer
spacer = Spacer(length=0.35)

section_array = SectionArray(
    input_df,
    sagging_parameter=2_000,
    sagging_temperature=15,
    bundle_number=3,
    spacer=spacer,
)
```

If no `spacer` is provided, a default `Spacer()` is used automatically.

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

