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


# @pytest.mark.unit_test
# def test_plot_planes_with_mechaphlowers_helpers():
#     """
#     Test plotting functions: add_key_points_line_and_span and add_obstacle_plane_and_intersection.

#     This test recreates the exact workflow from the jupyter notebook cell to verify
#     that the plotting helpers work correctly with real data from a section geometry.

#     Args:
#         pt0: First support point (from conftest fixture)
#         pt1: Second support point (from conftest fixture)
#         obstacle: Obstacle point (from conftest fixture)
#         spans1: Span points between pt0 and pt1 (from conftest fixture)
#     """
#     # Setup: create figure and points array
#     key_points = np.vstack([pt0, pt1])
#     fig_section_mf = figure_factory("blank")

#     # Test 1: add_axis_points_line_and_span
#     add_axis_points_line_and_span(
#         fig_section_mf,
#         axis_points=key_points,
#         span_points=spans1,
#         axis_points_labels=["pt0", "pt1"],
#     )

#     # Verify figure has traces added
#     assert (
#         len(fig_section_mf.data) > 0
#     ), "Figure should have traces after add_axis_points_line_and_span"

#     # Test 2: add_obstacle_plane_and_intersection
#     intersections_mf, u_plane_mf, v_plane_mf = (
#         add_obstacle_plane_and_intersection(
#             fig_section_mf,
#             obstacle=obstacle,
#             key_points=key_points,
#             span_points=spans1,
#             plane_scale=150,
#             plane_grid_size=150,
#             fine_tuning=True,
#         )
#     )

#     # Verify returns
#     assert intersections_mf is not None, "Intersections should not be None"
#     assert u_plane_mf is not None, "u_plane should not be None"
#     assert v_plane_mf is not None, "v_plane should not be None"

#     # Verify intersection points
#     if len(intersections_mf) > 0:
#         assert (
#             intersections_mf.shape[1] == 3
#         ), "Each intersection point should have 3 coordinates"

#     # Verify plane basis vectors
#     assert u_plane_mf.shape == (3,), "u_plane should be a 3D vector"
#     assert v_plane_mf.shape == (3,), "v_plane should be a 3D vector"

#     # Test 3: Update layout and verify
#     fig_section_mf.update_layout(
#         title="Section Geometry: Line, Obstacle, Span, and Orthogonal Plane",
#         scene=dict(
#             xaxis_title="X (m)",
#             yaxis_title="Y (m)",
#             zaxis_title="Z (m)",
#             aspectmode="data",
#         ),
#         width=1000,
#         height=800,
#         showlegend=True,
#         legend=dict(x=0.02, y=0.98),
#     )

#     # Verify layout was updated
#     assert (
#         fig_section_mf.layout.title.text
#         == "Section Geometry: Line, Obstacle, Span, and Orthogonal Plane"
#     )
#     assert fig_section_mf.layout.scene.xaxis.title.text == "X (m)"
#     assert fig_section_mf.layout.scene.yaxis.title.text == "Y (m)"
#     assert fig_section_mf.layout.scene.zaxis.title.text == "Z (m)"
#     assert fig_section_mf.layout.scene.aspectmode == "data"
#     assert fig_section_mf.layout.width == 1000
#     assert fig_section_mf.layout.height == 800

#     # Verify figure is displayable (has data)
#     assert len(fig_section_mf.data) > 0, "Figure should have multiple traces"


# @pytest.mark.unit_test
# def test_point_distance_visualization():
#     """
#     Test the complete point_distance workflow with visualization.

#     This test demonstrates the end-to-end workflow of computing and visualizing
#     the distance from a point to a span, including:
#     - Line geometry between two supports
#     - Orthogonal plane
#     - Intersection points
#     - Distance vector and projections
#     """
#     key_points = np.vstack([pt0, pt1])
#     fig = figure_factory("blank")

#     # Step 1: Add axis points line and span
#     add_axis_points_line_and_span(
#         fig,
#         axis_points=key_points,
#         span_points=spans1,
#         axis_points_labels=["pt0", "pt1"],
#     )

#     assert len(fig.data) == 3, "Should have 3 traces: axis_points, line, span"
#     traces_before_plane = len(fig.data)

#     # Step 2: Add obstacle, plane, and intersection
#     intersections, u_plane, v_plane = add_obstacle_plane_and_intersection(
#         fig,
#         obstacle=obstacle,
#         axis_points=key_points,
#         span_points=spans1,
#         plane_scale=150,
#         plane_grid_size=150,
#         fine_tuning=True,
#     )

#     # Should have added obstacle, plane, and intersection traces
#     assert len(fig.data) > traces_before_plane, "Plane traces should be added"
#     assert intersections.size > 0, "Intersection points should be found"

#     # Step 3: Compute distance and projections
#     if len(intersections) > 0:
#         intersection_point = intersections[0]
#         line_direction = key_points[1] - key_points[0]
#         line_direction_normalized = line_direction / np.linalg.norm(
#             line_direction
#         )

#         # Compute distance and projections in plane
#         distance_3d, projection_u, projection_v = points_distance_inside_plane(
#             intersection_point,
#             obstacle,
#             plane_normal=line_direction,
#             line_direction_normalized=line_direction_normalized,
#             u_plane=u_plane,
#             v_plane=v_plane,
#         )

#         assert distance_3d > 0, "Distance should be positive"
#         assert isinstance(
#             projection_u, (float, np.floating)
#         ), "projection_u should be a float"
#         assert isinstance(
#             projection_v, (float, np.floating)
#         ), "projection_v should be a float"

#         # Step 4: Get projection points
#         u_projection_point, v_projection_point = get_projection_points(
#             obstacle, projection_u, projection_v, u_plane, v_plane
#         )

#         assert u_projection_point.shape == (
#             3,
#         ), "u_projection_point should be 3D"
#         assert v_projection_point.shape == (
#             3,
#         ), "v_projection_point should be 3D"

#         # Step 5: Add distance visualization
#         traces_before_distance = len(fig.data)

#         # Add distance arrow
#         fig.add_trace(
#             go.Scatter3d(
#                 x=[obstacle[0], intersection_point[0]],
#                 y=[obstacle[1], intersection_point[1]],
#                 z=[obstacle[2], intersection_point[2]],
#                 mode="lines",
#                 line=dict(color="black", width=4),
#                 name=f"Distance: {distance_3d:.2f} m",
#                 legendgroup="distance",
#             )
#         )

#         # Add distance label
#         midpoint = (obstacle + intersection_point) / 2
#         fig.add_trace(
#             go.Scatter3d(
#                 x=[midpoint[0]],
#                 y=[midpoint[1]],
#                 z=[midpoint[2]],
#                 mode="text",
#                 text=[f"{distance_3d:.2f} m"],
#                 textposition="top center",
#                 textfont=dict(size=14, color="black"),
#                 name="Distance Text",
#                 legendgroup="distance",
#             )
#         )

#         # Add projection arrows
#         fig.add_trace(
#             go.Scatter3d(
#                 x=[obstacle[0], u_projection_point[0]],
#                 y=[obstacle[1], u_projection_point[1]],
#                 z=[obstacle[2], u_projection_point[2]],
#                 mode="lines",
#                 line=dict(color="blue", width=3),
#                 name=f"U Distance: {abs(projection_u):.2f} m",
#                 legendgroup="projections",
#             )
#         )

#         fig.add_trace(
#             go.Scatter3d(
#                 x=[obstacle[0], v_projection_point[0]],
#                 y=[obstacle[1], v_projection_point[1]],
#                 z=[obstacle[2], v_projection_point[2]],
#                 mode="lines",
#                 line=dict(color="cyan", width=3),
#                 name=f"V Distance: {abs(projection_v):.2f} m",
#                 legendgroup="projections",
#             )
#         )

#         # Step 6: Verify figure has all traces
#         assert (
#             len(fig.data) == traces_before_distance + 4
#         ), "Should have added 4 distance-related traces"

#         # Step 7: Verify distance computation
#         # Pythagorean theorem: sqrt(u^2 + v^2) should equal distance_3d
#         projected_distance = np.sqrt(projection_u**2 + projection_v**2)
#         assert (
#             abs(projected_distance - distance_3d) < 1e-6
#         ), f"Projected distance ({projected_distance}) should equal 3D distance ({distance_3d})"

#     # Step 8: Update layout
#     fig.update_layout(
#         title="Point Distance Analysis",
#         scene=dict(
#             xaxis_title="X (m)",
#             yaxis_title="Y (m)",
#             zaxis_title="Z (m)",
#             aspectmode="data",
#         ),
#         width=1000,
#         height=800,
#         showlegend=True,
#         legend=dict(x=0.02, y=0.98),
#     )

#     # Verify final figure
#     assert fig.layout.title.text == "Point Distance Analysis"
#     assert len(fig.data) > 0, "Figure should have traces"
#     # fig.show()  # This will display the figure in an interactive environment


@pytest.mark.unit_test
def test_distance_engine():
    
    de = DistanceEngine()
    de.add_curves(spans1)
    de.add_span_frame(pt0, pt1)

    dr = de.plane_distance(obstacle, frame="span")
    
    fig = de.plot(distance_result=dr, show_plane=True, show_projections=True)
    
    fig.show()
    
    assert isinstance(dr, DistanceResult), "DistanceEngine should return a DistanceResult"
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
    plt_engine = PlotEngine.builder_from_balance_engine(balance_engine_angles)
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

    plt_engine.point_distance(
        fig=fig,
        span_index=0,
        point=obstacle_point,
        show_figure=False,  # Don't display during test
    )


    # Verify returns
    assert fig is not None, "Figure should be returned"
    assert isinstance(
        fig, go.Figure
    ), "Returned figure should be a Plotly Figure"

    # Verify figure has traces
    assert len(fig.data) > 0, "Figure should contain traces"

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
            show_figure=False,
        )
        assert fig is not None, "Second figure should be returned"
        assert (
            "Span 1" in fig.layout.title.text
        ), "Second title should reference span 1"

    # Test error handling: invalid span index
    with pytest.raises(IndexError):
        plt_engine.point_distance(
            span_index=999,  # Out of range
            point=obstacle_point,
            show_figure=False,
        )

    # Test error handling: invalid point shape
    with pytest.raises(ValueError):
        plt_engine.point_distance(
            span_index=0,
            point=np.array([250.0, 20.0]),  # Only 2D
            show_figure=False,
        )

    fig.show()  # Display the figure for visual verification (optional)
