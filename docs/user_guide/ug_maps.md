# Plotting maps

## Geographic coordinates system

Mechaphlowers enables to plot the line on a map.  
The feature is implemented in the core system but not available through the user API.



## Dash example

In order to have a full display and intercative map, user can use the follwing dash code.

```python 
import json
import os

import dash
import plotly.graph_objects as go
from dash import dcc, html

from mechaphlowers.plotting.geography.create_elevation_profile import (
    create_elevation_profile,
)
from mechaphlowers.plotting.geography.create_line_map import create_line_map
from mechaphlowers.plotting.geography.helpers import geo_info_from_gps
from mechaphlowers.plotting.geography.type_hints import SupportGeoInfo


def main():
    app = dash.Dash(__name__)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(
        os.path.join(
            current_dir,
            "...[to be replaced] data/geography/example_supports_gps_points.json",
        ),
        "r",
    ) as f:
        data = json.load(f)
        markers: list[SupportGeoInfo] = data

    geo_points = geo_info_from_gps(markers)

    line_map_fig = go.Figure()
    create_line_map(line_map_fig, geo_points)

    elevation_profile_fig = go.Figure()
    create_elevation_profile(elevation_profile_fig, geo_points)

    app.layout = html.Div(
        [
            dcc.Graph(
                id='line_map',
                figure=line_map_fig,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': [
                        'pan2d',
                        'select2d',
                        'lasso2d',
                        'autoScale2d',
                    ],
                },
            ),
            dcc.Graph(
                id="elevation_profile",
                figure=elevation_profile_fig,
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "pan2d",
                        "select2d",
                        "lasso2d",
                        "autoScale2d",
                    ],
                },
            ),
        ],
        style={
            "width": "100%",
            "height": "100%",
            "display": "flex",
            "flexDirection": "column",
            "justifyContent": "center",
            "alignItems": "center",
        },
    )
    app.run(debug=True)


if __name__ == "__main__":
    main()
```

