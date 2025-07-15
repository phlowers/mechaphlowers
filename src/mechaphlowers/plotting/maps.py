# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import plotly.graph_objects as go

from mechaphlowers.entities.geography import (
    SupportGeoInfo,
)


def plot_line_map(fig: go.Figure, supports_geo_info: SupportGeoInfo) -> None:
    """
    Create a line map from support geo info.
    Args:
        fig: The figure to add the line map to.
        supports_geo_info: Support geo info containing arrays of data for multiple supports.
    """
    # Extract arrays from the SupportGeoInfo object
    gps_lat = supports_geo_info["latitude"]
    gps_lon = supports_geo_info["longitude"]
    elevations = supports_geo_info["elevation"]
    distances_to_next = supports_geo_info["distance_to_next"]
    bearings_to_next = supports_geo_info["bearing_to_next"]
    directions_to_next = supports_geo_info["direction_to_next"]
    lambert_x, lambert_y = supports_geo_info["lambert_93"]

    if len(gps_lat) == 0:
        return

    # Create hover text for each point
    hover_texts = []
    for i in range(len(gps_lat)):
        # Handle the last support differently since it doesn't have a "next" support
        if i < len(distances_to_next):
            distance_info = (
                f"<br>Distance to next: {distances_to_next[i]:.2f} km"
            )
            bearing_info = f"<br>Bearing to next: {bearings_to_next[i]:.1f}° ({directions_to_next[i]})"
        else:
            distance_info = "<br>Distance to next: N/A (last support)"
            bearing_info = "<br>Bearing to next: N/A (last support)"

        elevation_info = f"<br>Elevation: {elevations[i]:.2f} m"
        lambert_info = f"<br>Lambert X: {lambert_x[i]:.2f}<br>Lambert Y: {lambert_y[i]:.2f}"

        hover_text = (
            f"<b>Support {i}</b><br>"
            f"Lat: {gps_lat[i]:.6f}<br>"
            f"Lon: {gps_lon[i]:.6f}"
            + distance_info
            + bearing_info
            + elevation_info
            + lambert_info
        )
        hover_texts.append(hover_text)

    # Create combined lines+markers trace
    combined_trace = go.Scattermapbox(
        lat=gps_lat.tolist(),
        lon=gps_lon.tolist(),
        mode="lines+markers",
        line={"color": "#666666", "width": 2},
        marker=go.scattermapbox.Marker(size=10, color="#FF0000", opacity=0.8),
        text=hover_texts,
        hoverinfo="text",
        hovertemplate='%{text}<extra></extra>',
        name="Power Line Supports",
        showlegend=True,
    )

    fig.add_trace(combined_trace)

    center_lat = gps_lat.mean()
    center_lon = gps_lon.mean()

    fig.update_layout(
        mapbox={
            "style": "carto-positron",
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": 11,
        },
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=700,
        width=1200,
        showlegend=True,
    )
