<img src="_static/logos/mechaphlowers_fullsize.png" width="200" height="200" alt="Phlowers logo" style="float: right; display: block; margin: 0 auto"/>

# Mechaphlowers


Mechaphlowers is a user oriented package dedicated to mechanical and geometrical calculations for overhead power lines.  
Mechaphlowers is part of the [phlowers](https://phlowers.readthedocs.io/en/latest/) project.

## Features

- loading simplified span referenced data of a section.
- 3D plot of the section.

<!-- ## Why use mechaphlowers ? -->


## Examples

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import mechaphlowers as mph


# load data
from mechaphlowers.data.catalog import section_factory_sample_data

section_array = mph.SectionArray(
    pd.DataFrame(
        section_factory_sample_data(6,2)
    )
)
section_array.sagging_parameter = 2000
section_array.sagging_temperature = 15

# Load cable from catalog
from mechaphlowers.data.catalog import sample_cable_catalog
cable_array_AM600 = sample_cable_catalog.get_as_object(["ASTER600"])

# Create balance engine and plot engine
engine = mph.BalanceEngine(cable_array=cable_array_AM600, section_array=section_array)
plt = mph.PlotEngine.builder_from_balance_engine(engine)

# initialize plotly figure
fig = mph.plotting.figure_factory(context="std")
# you can also do
# fig = go.Figure()

# Chain your changes and preview for printing in figure
engine.solve_adjustment()
engine.solve_change_state(new_temperature=15.)
plt.preview_line3d(fig)

engine.solve_change_state(new_temperature=90.)
plt.preview_line3d(fig)

engine.solve_change_state(new_temperature=15., wind_pressure=np.array([0, 0, 0, 500, 500, 500]))
plt.preview_line3d(fig)

# plot the result
fig.show()
```
--8<-- "docs/user_guide/assets/result_example0.html"

