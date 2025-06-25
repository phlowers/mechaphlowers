# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import plotly.graph_objects as go

from mechaphlowers.plotting.geography.type_hints import (
    SupportGeoInfo,
)


def create_elevation_profile(
    fig: go.Figure, supports_geo_info: list[SupportGeoInfo]
) -> None:
    """
    Create an elevation profile plot from a list of support geo info.
    Args:
        fig: The figure to add the elevation profile to.
        supports_geo_info: A list of support geo info.
    """

    cumulative_distance: list[float] = [0]  # Start at 0
    elevations = [support["elevation"] for support in supports_geo_info]

    for support in supports_geo_info:
        cumulative_distance.append(
            cumulative_distance[-1] + support["distance_to_next"]
        )

    fig.add_trace(
        go.Scatter(
            x=cumulative_distance,
            y=elevations,
            mode="lines+markers",
            name="Elevation Profile",
            line={"color": "#1f77b4", "width": 2},
            marker={"size": 8, "color": "#1f77b4"},
            hovertemplate="Marker: %{customdata}<br>Distance: %{x:.2f} km<br>Elevation: %{y:.2f} m<extra></extra>",
            customdata=list(range(len(supports_geo_info))),  # type: ignore
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
