# Input data format

This page describes the format of the input data needed to perform calculations about a line section in mechaphlowers.

!!! Units

    Unless specified otherwise, all units will be those of the International System of Units (SI).
    A list of these units can be found [here](https://en.wikipedia.org/wiki/International_System_of_Units).

In mechaphlowers a line section is described by following data:

- a sagging parameter
- a sagging temperature (in Celsius degrees)
- for each support:
    - a boolean named `suspension` describing whether it's a suspension or tension support
    - the conductor attachment altitude
    - the crossarm length
    - the line angle (in degrees)
    - the insulator length
- for each span:
    - the length of the span.

!!! important

    For now, sagging parameter and temperature are assumed to be **the same for each span** in a line section - which doesn't necessarily reflect reality.
    Support for disparate sagging parameters and temperatures may be added later.

Input data should be organized in a table (for example a pandas dataframe), where each row describes one support with its following span, except the last row which only describes the last support (since it doesn't have a "following" span). Hence the last span length is expected to be "not a number", typically `numpy.nan`.

For example, a line section could be described by following table:

|name|suspension|conductor_attachment_altitude|crossarm_length|line_angle|insulator_length|span_length|
|---------------|------|----|---|--|--|------|
|first support  |False |1     |12   |0  |0 |500   |
|second support |True  |-0.75 |10   |11 |4 |800   |
|third support  |False |0.5   |11.5 |0  |0 |      |

!!! Altitude

    Since the conductor attachment altitude is measured from the sea level, it may be negative.

In this example the span length between the first and second supports is 500m, and the span length between the second and third support is 800m. The line angle in the middle of the section is of 10 degrees.

You may use following code to define this data and load it so that it can be used by mechaphlowers:

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
