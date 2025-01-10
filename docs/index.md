<img src="_static/logos/mechaphlowers_fullsize.png" width="200" height="200" alt="Phlowers logo" style="float: right; display: block; margin: 0 auto"/>

# Mechaphlowers


Mechaphlowers is a user oriented package dedicated to mechanical calculus for overhead power lines.

## Features

- loading simplified span referenced data of a section.
- 3D plot of the section.

<!-- ## Why use mechaphlowers ? -->


## Examples

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mechaphlowers.entities import SectionDataFrame
from mechaphlowers.entities.arrays import SectionArray


# load data 
data = {
	"name": ["1", "2", "three", "support 4"],
	"suspension": [False, True, True, False],
	"conductor_attachment_altitude": [50.0, 40.0, 20.0, 10.0],
	"crossarm_length": [5.0,]* 4,
	"line_angle": [0.]* 4,
	"insulator_length": [0, 4, 3.2, 0],
	"span_length": [100, 200, 300, np.nan],
}

section = SectionArray(data=pd.DataFrame(data))

# Provide section to SectionDataFrame
frame = SectionDataFrame(section)

# set sagging parameter and temperatur 
section.sagging_parameter = 500
section.sagging_temperature = 15

# Display figure
fig = go.Figure()
frame.plot.line3d(fig)
fig.show()

# Reset figure
fig._data = []

# display only first span
frame.select(["1", "2"]).plot.line3d(fig)
fig.show()

```
