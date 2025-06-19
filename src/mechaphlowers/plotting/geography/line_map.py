# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import plotly.graph_objects as go

from mechaphlowers.plotting.geography.type_hints import SupportGeoInfo


def create_line_map(fig: go.Figure, markers: list[SupportGeoInfo]) -> None:
    """
    Create a line map from a list of support geo info.
    """
    marker_traces: list[go.Scattermapbox] = []


    for i, marker in enumerate(markers):
        # Calculate distance and bearing to next point if it exists
        distance_info = ""
        bearing_info = ""
        elevation_info = ""
        distance_info = f"<br>Distance to next: {marker['distance_to_next']:.2f} km"
        bearing_info = f"<br>Bearing to next: {marker['bearing_to_next']:.1f}° ({marker['direction_to_next']})"
        elevation_info = f"<br>Elevation: {marker['elevation']:.2f} m"

        
        marker_traces.append(
            go.Scattermapbox(
                lat=[marker['gps']['lat']],
                lon=[marker['gps']['lon']],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=10,
                    color='#FF0000',
                    opacity=0.8
                ),
                text=[i],
                hoverinfo="text",
                hovertemplate='<b>%{text}</b><br>' + 
                            'Lat: %{lat:.6f}<br>' + 
                            'Lon: %{lon:.6f}' +
                            distance_info +
                            bearing_info + 
                            elevation_info +
                            f"<br>Lambert X: {marker['lambert_93']['x']:.2f}<br>Lambert Y: {marker['lambert_93']['y']:.2f}"
                            '<extra></extra>',
                name=f"Marker {i} - {marker['gps']['lat']}-{marker['gps']['lon']}"
            )
        )

    # Create line trace connecting markers
    line_trace = go.Scattermapbox(
        lat=[marker['gps']['lat'] for marker in markers],
        lon=[marker['gps']['lon'] for marker in markers],
        mode='lines',
        line=dict(
            color='#666666',
            width=2
        ),
        hoverinfo='skip',
        showlegend=False
    )

    # Combine all traces
    traces = marker_traces + [line_trace]

    # Create the figure
    fig.add_traces(traces)

    # Update the layout
    fig.update_layout(
        mapbox=dict(
            style='carto-positron',
            center=dict(
                lat=42.19476145,
                lon=8.80258205
            ),
            zoom=11
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        height=700,
        width=1200,
        showlegend=False
    )


