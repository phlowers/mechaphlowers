"""
Tests for plotting plane and distance visualization functions.

This test module uses fixtures from the notebook distance_poc.ipynb 
to validate the plotting helper functions for geometric distance calculations.
"""

import numpy as np
import pytest
from mechaphlowers.plotting.distances import (
    add_key_points_line_and_span,
    add_obstacle_plane_and_intersection,
)
from mechaphlowers.plotting.plot import figure_factory


pt0 = np.array([10.0, 20.0, 30.0])
pt1 = np.array([40.0, 60.0, 30.0])
obstacle = np.array([25.0, 40.0, 50.0])
spans1 = 


@pytest.mark.unit_test
def test_plot_planes_with_mechaphlowers_helpers(pt0, pt1, obstacle, spans1):
    """
    Test plotting functions: add_key_points_line_and_span and add_obstacle_plane_and_intersection.
    
    This test recreates the exact workflow from the jupyter notebook cell to verify 
    that the plotting helpers work correctly with real data from a section geometry.
    
    Args:
        pt0: First support point (from conftest fixture)
        pt1: Second support point (from conftest fixture)
        obstacle: Obstacle point (from conftest fixture)
        spans1: Span points between pt0 and pt1 (from conftest fixture)
    """
    # Setup: create figure and points array
    key_points = np.vstack([pt0, pt1])
    fig_section_mf = figure_factory("blank")
    
    # Test 1: add_key_points_line_and_span
    add_key_points_line_and_span(
        fig_section_mf,
        key_points=key_points,
        span_points=spans1,
        key_points_labels=["pt0", "pt1"],
    )
    
    # Verify figure has traces added
    assert len(fig_section_mf.data) > 0, "Figure should have traces after add_key_points_line_and_span"
    
    # Test 2: add_obstacle_plane_and_intersection
    intersections_mf, u_plane_mf, v_plane_mf = add_obstacle_plane_and_intersection(
        fig_section_mf,
        obstacle=obstacle,
        key_points=key_points,
        span_points=spans1,
        plane_scale=150,
        plane_grid_size=150,
        fine_tuning=True
    )
    
    # Verify returns
    assert intersections_mf is not None, "Intersections should not be None"
    assert u_plane_mf is not None, "u_plane should not be None"
    assert v_plane_mf is not None, "v_plane should not be None"
    
    # Verify intersection points
    if len(intersections_mf) > 0:
        assert intersections_mf.shape[1] == 3, "Each intersection point should have 3 coordinates"
    
    # Verify plane basis vectors
    assert u_plane_mf.shape == (3,), "u_plane should be a 3D vector"
    assert v_plane_mf.shape == (3,), "v_plane should be a 3D vector"
    
    # Test 3: Update layout and verify
    fig_section_mf.update_layout(
        title="Section Geometry: Line, Obstacle, Span, and Orthogonal Plane",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )
    
    # Verify layout was updated
    assert fig_section_mf.layout.title.text == "Section Geometry: Line, Obstacle, Span, and Orthogonal Plane"
    assert fig_section_mf.layout.scene.xaxis.title.text == "X (m)"
    assert fig_section_mf.layout.scene.yaxis.title.text == "Y (m)"
    assert fig_section_mf.layout.scene.zaxis.title.text == "Z (m)"
    assert fig_section_mf.layout.scene.aspectmode == "data"
    assert fig_section_mf.layout.width == 1000
    assert fig_section_mf.layout.height == 800
    
    # Verify figure is displayable (has data)
    assert len(fig_section_mf.data) > 0, "Figure should have multiple traces"
    fig_section_mf.show()  # This will display the figure in an interactive environment
