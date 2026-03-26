# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Tests for plotting plane and distance visualization functions.

This test module uses fixtures from the notebook distance_poc.ipynb
to validate the plotting helper functions for geometric distance calculations.
"""

import numpy as np
import plotly.graph_objects as go
import pytest

from mechaphlowers.core.geometry.distances import (
    DistanceEngine,
    DistanceResult,
)
from mechaphlowers.plotting.plot import PlotEngine
from mechaphlowers.plotting.plot_distances import plot_distance_engine
from test.conftest import show_figures

pt0 = np.array([0.0, 0.0, 38.69021545])
pt1 = np.array([597.90765581, -200.92768965, 10.27928764])
obstacle = np.array([250, 20, 35])
spans1 = np.array(
    [
        [7.4562539, -4.37219131, 64.72062689],
        [27.10943395, -9.13999987, 58.88505574],
        [46.81335925, -14.05891168, 53.41758266],
        [66.56779145, -19.12821708, 48.31647882],
        [86.37250859, -24.34725505, 43.58013398],
        [106.22730494, -29.71541308, 39.20705599],
        [126.13199102, -35.23212691, 35.19587028],
        [146.08639349, -40.89688038, 31.54531945],
        [166.09035509, -46.70920525, 28.25426285],
        [186.14373462, -52.66868107, 25.3216762],
        [206.24640687, -58.77493499, 22.74665126],
        [226.3982626, -65.02764172, 20.52839554],
        [246.59920847, -71.42652335, 18.66623207],
        [266.84916706, -77.97134929, 17.15959914],
        [287.14807678, -84.66193623, 16.00805012],
        [307.49589191, -91.49814799, 15.21125335],
        [327.89258258, -98.47989559, 14.76899201],
        [348.3381347, -105.60713713, 14.68116404],
        [368.83255002, -112.87987781, 14.94778213],
        [389.37584612, -120.29816995, 15.56897372],
        [409.96805639, -127.86211297, 16.54498102],
        [430.60923004, -135.57185346, 17.8761611],
        [451.29943217, -143.4275852, 19.56298602],
        [472.0387437, -151.42954921, 21.60604297],
        [492.82726151, -159.57803388, 24.00603446],
        [513.66509834, -167.87337503, 26.76377855],
        [534.55238296, -176.315956, 29.88020911],
        [555.48926011, -184.9062078, 33.35637613],
        [576.47589059, -193.64460927, 37.19344604],
    ]
)


@pytest.mark.unit_test
def test_distance_engine():
    de = DistanceEngine()
    de.add_curves(spans1)
    de.add_span_frame(pt0, pt1)

    dr = de.plane_distance(obstacle, frame="span")

    fig = plot_distance_engine(de, distance_result=dr, show_plane=True, show_projections=True)
    if show_figures:
        fig.show()

    assert isinstance(
        dr, DistanceResult
    ), "DistanceEngine should return a DistanceResult"
    assert dr.distance_3d > 0, "Distance should be positive"


@pytest.mark.unit_test
def test_point_distance_method_with_plot_engine(balance_engine_angles):
    """
    Test the PlotEngine.point_distance() method inspired by test_plot_3d_sandbox.

    This test verifies that the point_distance method correctly computes and
    visualizes the distance from a point (obstacle) to a span in a real power
    line section model.

    Args:
        balance_engine_angles: Balance engine fixture with angle configuration.
    """
    # Setup: Create PlotEngine from balance engine
    plt_engine = PlotEngine(balance_engine_angles)
    balance_engine_angles.solve_adjustment()
    balance_engine_angles.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1, 1])
    )

    # Solve with wind and ice loads
    balance_engine_angles.solve_change_state(
        wind_pressure=np.array([1, 1, 1, np.nan]) * 200,
    )

    # Define an obstacle point to analyze
    obstacle_point = np.array([250.0, 20.0, 35.0])
    fig = go.Figure()

    plt_engine.preview_line3d(fig=fig)

    # Test: Call point_distance method for span 0

    dr = plt_engine.point_distance(
        fig=fig,
        span_index=0,
        point=obstacle_point,
    )

    # Verify returns
    assert isinstance(
        dr, DistanceResult
    ), "Returned object should be a DistanceResult instance"

    # Verify figure has traces

    # Verify figure layout
    assert "Span 0" in fig.layout.title.text, "Title should contain span index"
    assert fig.layout.scene.xaxis.title.text == "X (m)"
    assert fig.layout.scene.yaxis.title.text == "Y (m)"
    assert fig.layout.scene.zaxis.title.text == "Z (m)"

    # Verify span points are valid
    supports = plt_engine.get_supports_points()
    assert len(supports) >= 2, "Should have at least 2 support points"

    # Test different span index (if available)
    if len(supports) > 2:
        plt_engine.point_distance(
            fig=fig,
            span_index=1,
            point=np.array([300.0, -40.0, 40.0]),
        )
        assert (
            "Span 1" in fig.layout.title.text
        ), "Second title should reference span 1"

    # Test error handling: invalid span index
    with pytest.raises(IndexError):
        plt_engine.point_distance(
            span_index=999,  # Out of range
            point=obstacle_point,
        )

    # Test error handling: invalid point shape
    with pytest.raises(ValueError):
        plt_engine.point_distance(
            span_index=0,
            point=np.array([250.0, 20.0]),  # Only 2D
        )
    if show_figures:
        fig.show()  # Display the figure for visual verification (optional)
