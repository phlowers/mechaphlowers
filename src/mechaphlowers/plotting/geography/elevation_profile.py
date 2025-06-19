# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import plotly.graph_objects as go

from mechaphlowers.plotting.geography.type_hints import SupportGeoInfo


def create_elevation_profile(fig: go.Figure, markers: list[SupportGeoInfo]) -> None:
    """
    Create an elevation profile plot from a list of support geo info.
    """
    
    # Calculate cumulative distances and prepare elevation data
    cumulative_distance: list[float] = [0]  # Start at 0
    elevations = [markers[0]['elevation']]  # First elevation
    lambert_93 = []  # Store Lambert 93 coordinates
    
    for marker in markers:
        cumulative_distance.append(cumulative_distance[-1] + marker['distance_to_next'])
        elevations.append(marker['elevation'])
        lambert_93.append(marker['lambert_93'])

    # Create the figure
    
    customdata = list(zip(range(len(markers)), lambert_93))
    # Add the main line plot
    fig.add_trace(go.Scatter(
        x=cumulative_distance,
        y=elevations,
        mode='lines+markers',
        name='Elevation Profile',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8, color='#1f77b4'),
        hovertemplate='Marker: %{customdata[0]}<br>Distance: %{x:.2f} km<br>Elevation: %{y:.2f} m<br>Lambert X: %{customdata[1][0]:.2f}<br>Lambert Y: %{customdata[1][1]:.2f}<extra></extra>',
        customdata=customdata,  # type: ignore
    ))

    # Update layout
    fig.update_layout(
        title='Elevation Profile of Power Line',
        xaxis_title='Cumulative Distance (km)',
        yaxis_title='Elevation (m)',
        width=1200,
        height=500,
        showlegend=False,
        hovermode='closest',
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
    )

