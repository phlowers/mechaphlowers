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
    - a boolean named `suspension` describing whether it's a suspension or tension support
    - the conductor attachment altitude
    - the crossarm length
    - the line angle (in degrees)
    - the insulator length
- for each span:
    - the span length, denoted later as $a$.

!!! important

    For now, sagging parameter and temperature are assumed to be **the same for each span** in a line section - which doesn't necessarily reflect reality.
    Support for disparate sagging parameters and temperatures may be added later.

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
    })
    section_array = SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)
    print(section_array)

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

External loads can be added to the cable array. We only support wind loads, but it's easy to extend this by adding more external loads. To add a weather load, you need to create an instance of `WeatherArray` class, and pass it to the SectionDataFrame.
The following example shows how to add a wind load on the cable.

!!! important

	The ice_thickness and wind_pressure are in cm and Pa respectively.
	The format of those vectors is span oriented: their size is the same than the section but the last value is not used 
	That's why we put `np.nan` at the end.

Then you can display the effect of this load with:

- load_angle
- resulting_norm
- load_coefficient
- ice_load
- wind_load

```python
# Define data
weather = WeatherArray(
	pd.DataFrame(
		{
			"ice_thickness": [1, 2.1, 0.0, np.nan],
			"wind_pressure": [240.12, 0.0, 12.0, np.nan],
		}
	)
)

# add weather
frame.add_weather(weather=weather)

# get effects of the load: example with the load_angle
frame.cable_loads.load_angle
```

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

