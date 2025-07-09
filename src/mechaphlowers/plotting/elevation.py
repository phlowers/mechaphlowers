# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import plotly.graph_objects as go

from mechaphlowers.entities.geography import (
    SupportGeoInfo,
)


def plot_elevation_profile(
    fig: go.Figure, supports_geo_info: SupportGeoInfo
) -> None:
    """
    Create an elevation profile plot from support geo info.
    Args:
        fig: The figure to add the elevation profile to.
        support_geo_info: Support geo info containing elevation and distance data.
    """

    # Extract arrays from the SupportGeoInfo object
    elevations = supports_geo_info["elevation"]
    distances = supports_geo_info["distance_to_next"]

    # Calculate cumulative distance
    cumulative_distance = np.cumsum(distances)

    fig.add_trace(
        go.Scatter(
            x=cumulative_distance,
            y=elevations,
            mode="lines+markers",
            name="Elevation Profile",
            line={"color": "#1f77b4", "width": 2},
            marker={"size": 8, "color": "#1f77b4"},
            hovertemplate="Marker: %{customdata}<br>Distance: %{x:.2f} km<br>Elevation: %{y:.2f} m<extra></extra>",
            customdata=list(range(len(elevations))),  # type: ignore
        )
    )

    fig.update_layout(
        title="Elevation Profile of Power Line",
        xaxis_title="Cumulative Distance (km)",
        yaxis_title="Elevation (m)",
        width=1200,
        height=500,
        showlegend=False,
        hovermode="closest",
        margin={"l": 50, "r": 50, "t": 50, "b": 50},
        xaxis={"showgrid": True, "gridwidth": 1, "gridcolor": "LightGray"},
        yaxis={"showgrid": True, "gridwidth": 1, "gridcolor": "LightGray"},
    )
