# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import json
import os

import dash
import plotly.graph_objects as go
from dash import dcc, html

from mechaphlowers.plotting.geography.elevation_profile import (
    create_elevation_profile,
)
from mechaphlowers.plotting.geography.line_map import create_line_map
from mechaphlowers.plotting.geography.type_hints import SupportGeoInfo


def main():
    app = dash.Dash(__name__)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "../../data/geography/example_supports.json"), "r") as f:
        data = json.load(f) 
        markers: list[SupportGeoInfo] = data
        
    line_map_fig = go.Figure()
    create_line_map(line_map_fig, markers)
    
    elevation_profile_fig = go.Figure()
    create_elevation_profile(elevation_profile_fig, markers)    

    
    app.layout = html.Div([
        dcc.Graph(
            id='line_map',
            figure=line_map_fig,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d']
            }
        ),
        dcc.Graph(
            id='elevation_profile',
            figure=elevation_profile_fig,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d']
            }
        )
    ], style={'width': '100%', 'height': '100%', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center'})
    app.run(debug=True)

if __name__ == '__main__':
    main()