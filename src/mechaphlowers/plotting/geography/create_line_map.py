# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import plotly.graph_objects as go

from mechaphlowers.plotting.geography.type_hints import (
    SupportGeoInfo,
)


def create_line_map(
    fig: go.Figure, supports_geo_info: list[SupportGeoInfo]
) -> None:
    """
    Create a line map from a list of support geo info.
    Args:
        fig: The figure to add the line map to.
        supports_geo_info: A list of support geo info.
    """
    if not supports_geo_info:
        return

    markers: list[go.Scattermapbox] = []

    for i, support in enumerate(supports_geo_info):
        distance_info = (
            f"<br>Distance to next: {support['distance_to_next']:.2f} km"
        )
        bearing_info = f"<br>Bearing to next: {support['bearing_to_next']:.1f}° ({support['direction_to_next']})"
        elevation_info = f"<br>Elevation: {support['elevation']:.2f} m"

        markers.append(
            go.Scattermapbox(
                lat=[support["gps"]["lat"]],
                lon=[support["gps"]["lon"]],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=10, color='#FF0000', opacity=0.8
                ),
                text=[i],
                hoverinfo="text",
                hovertemplate='<b>%{text}</b><br>'
                + 'Lat: %{lat:.6f}<br>'
                + 'Lon: %{lon:.6f}'
                + distance_info
                + bearing_info
                + elevation_info
                + f"<br>Lambert X: {support['lambert_93']['x']:.2f}<br>Lambert Y: {support['lambert_93']['y']:.2f}"
                '<extra></extra>',
                name=f"Support {i} - {support['gps']['lat']}-{support['gps']['lon']}",
            )
        )

    line_trace = go.Scattermapbox(
        lat=[support["gps"]["lat"] for support in supports_geo_info],
        lon=[support["gps"]["lon"] for support in supports_geo_info],
        mode='lines',
        line={"color": "#666666", "width": 2},
        hoverinfo='skip',
        showlegend=False,
    )

    traces = markers + [line_trace]

    fig.add_traces(traces)

    center_lat = sum(
        [support["gps"]["lat"] for support in supports_geo_info]
    ) / len(supports_geo_info)
    center_lon = sum(
        [support["gps"]["lon"] for support in supports_geo_info]
    ) / len(supports_geo_info)

    fig.update_layout(
        mapbox={
            "style": "carto-positron",
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": 11,
        },
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=700,
        width=1200,
        showlegend=False,
    )
