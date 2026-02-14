"""Plotting helpers for distance and plane visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, NamedTuple

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]

from mechaphlowers.core.geometry.planes import (
    meshgrid_plane,
)

if TYPE_CHECKING:
    from mechaphlowers.core.geometry.distances import (
        DistanceEngine,
        DistanceResult,
    )


# ============================================================================
# Configuration Classes
# ============================================================================


class KeyPointsStyle(NamedTuple):
    """Styling configuration for key points visualization."""

    colors: str = "darkviolet"
    size: float = 8.0
    text_position: str = "top center"


class LineStyle(NamedTuple):
    """Styling configuration for line visualization."""

    name: str = "Key points line"
    color: str = "green"
    width: float = 4.0
    dash: str = "dash"


class SpanStyle(NamedTuple):
    """Styling configuration for span points visualization."""

    name: str = "Span"
    color: str = "darkblue"
    marker_size: float = 3.0
    show_legend: bool = False


class AxisPointsLineConfig(NamedTuple):
    """Configuration for axis points and line visualization."""

    axis_points_style: KeyPointsStyle = KeyPointsStyle()
    line_style: LineStyle = LineStyle()
    show_axis_points: bool = True


class KeyPointsLineConfig(NamedTuple):
    """Complete configuration for key points, line, and span visualization."""

    axis_points_style: KeyPointsStyle = KeyPointsStyle()
    line_style: LineStyle = LineStyle()
    span_style: SpanStyle = SpanStyle()


@dataclass
class ObstacleStyle:
    """Styling configuration for obstacle visualization."""

    name: str = "Obstacle"
    color: str = "red"
    size: float = 12.0
    symbol: str = "cross"
    text_position: str = "top center"


@dataclass
class PlaneStyle:
    """Styling configuration for plane visualization."""

    name: str = "Orthogonal Plane"
    colorscale: str = "Greens"
    opacity: float = 0.2


@dataclass
class IntersectionStyle:
    """Styling configuration for intersection points visualization."""

    name: str = "Intersection"
    color: str = "purple"
    size: float = 4.0
    text_position: str = "top center"


@dataclass
class ObstaclePlaneConfig:
    """Complete configuration for obstacle, plane, and intersection visualization."""

    plane_scale: float = 10.0
    plane_grid_size: int = 10
    fine_tuning: bool = False
    obstacle_style: ObstacleStyle = None  # type: ignore[assignment]
    plane_style: PlaneStyle = None  # type: ignore[assignment]
    intersection_style: IntersectionStyle = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Initialize default style objects if not provided."""
        if self.obstacle_style is None:
            self.obstacle_style = ObstacleStyle()
        if self.plane_style is None:
            self.plane_style = PlaneStyle()
        if self.intersection_style is None:
            self.intersection_style = IntersectionStyle()


# ============================================================================
# Validation Helpers
# ============================================================================


def _validate_axis_points(axis_points: np.ndarray) -> None:
    """Validate that axis_points is a 2x3 array."""
    if axis_points.shape != (2, 3):
        raise ValueError(
            "axis_points must be an array of shape (2, 3) for the two key points"
        )


def _validate_span_points(span_points: np.ndarray) -> None:
    """Validate that span_points is an Nx3 array."""
    if span_points.ndim != 2 or span_points.shape[1] != 3:
        raise ValueError("span_points must be an array of shape (N, 3)")


def _validate_obstacle(obstacle: np.ndarray) -> None:
    """Validate that obstacle is a 1D array of shape (3,)."""
    if obstacle.shape != (3,):
        raise ValueError("obstacle must be a 1D array of shape (3,)")


# ============================================================================
# Trace Creation Helpers
# ============================================================================


def _add_axis_points_trace(
    fig: go.Figure,
    axis_points: np.ndarray,
    labels: Iterable[str],
    style: KeyPointsStyle,
) -> None:
    """Add key points marker trace to figure."""
    fig.add_trace(
        go.Scatter3d(
            x=axis_points[:, 0],
            y=axis_points[:, 1],
            z=axis_points[:, 2],
            mode="markers+text",
            marker=dict(size=style.size, color=list(style.colors)),
            text=list(labels),
            textposition=style.text_position,
            name="Span X axis",
            legendgroup="points",
        )
    )


def _add_line_trace(
    fig: go.Figure,
    axis_points: np.ndarray,
    style: LineStyle,
) -> None:
    """Add line trace connecting key points to figure."""
    fig.add_trace(
        go.Scatter3d(
            x=axis_points[:, 0],
            y=axis_points[:, 1],
            z=axis_points[:, 2],
            mode="lines",
            line=dict(color=style.color, width=style.width, dash=style.dash),
            name=style.name,
            legendgroup="line",
        )
    )


def _add_span_trace(
    fig: go.Figure,
    span_points: np.ndarray,
    style: SpanStyle,
) -> None:
    """Add span points marker trace to figure."""
    fig.add_trace(
        go.Scatter3d(
            x=span_points[:, 0],
            y=span_points[:, 1],
            z=span_points[:, 2],
            mode="markers",
            marker=dict(color=style.color, size=style.marker_size),
            name=style.name,
            showlegend=style.show_legend,
            legendgroup="span",
        )
    )


def _add_obstacle_trace(
    fig: go.Figure,
    obstacle: np.ndarray,
    style: ObstacleStyle,
) -> None:
    """Add obstacle point trace to figure."""
    fig.add_trace(
        go.Scatter3d(
            x=[obstacle[0]],
            y=[obstacle[1]],
            z=[obstacle[2]],
            mode="markers+text",
            marker=dict(
                size=style.size,
                color=style.color,
                symbol=style.symbol,
            ),
            text=[style.name],
            textposition=style.text_position,
            name=style.name,
            legendgroup="obstacle",
        )
    )


def _add_plane_surface_trace(
    fig: go.Figure,
    x_plane: np.ndarray,
    y_plane: np.ndarray,
    z_plane: np.ndarray,
    style: PlaneStyle,
) -> None:
    """Add plane surface trace to figure."""
    fig.add_trace(
        go.Surface(
            x=x_plane,
            y=y_plane,
            z=z_plane,
            opacity=style.opacity,
            colorscale=style.colorscale,
            name=style.name,
            showscale=False,
        )
    )


def _add_intersection_trace(
    fig: go.Figure,
    intersections: np.ndarray,
    style: IntersectionStyle,
) -> None:
    """Add intersection points trace to figure if intersections exist."""
    if intersections.size == 0:
        return

    fig.add_trace(
        go.Scatter3d(
            x=intersections[:, 0],
            y=intersections[:, 1],
            z=intersections[:, 2],
            mode="markers+text",
            marker=dict(size=style.size, color=style.color),
            text=[style.name] * intersections.shape[0],
            textposition=style.text_position,
            name=style.name,
            legendgroup="intersection",
        )
    )


# ============================================================================
# Public API - Main Functions
# ============================================================================


def add_axis_points_line(
    fig: go.Figure,
    axis_points: np.ndarray,
    *,
    axis_points_labels: Iterable[str] | None = None,
    show_axis_points: bool = True,
    axis_points_colors: tuple[str, str] | None = None,
    axis_points_size: float | None = None,
    line_name: str | None = None,
    line_color: str | None = None,
    line_width: float | None = None,
    line_dash: str | None = None,
    config: AxisPointsLineConfig | None = None,
) -> None:
    """Add axis points and connecting line to a figure.

    This function adds two traces to the figure:
    1. Axis points as markers with text labels (optional)
    2. A line connecting the axis points (always plotted)

    Args:
        fig: Plotly figure to update.
        axis_points: Array of shape (2, 3) containing the axis points.
        axis_points_labels: Optional labels for axis points. Defaults to ["pt0", "pt1"].
        show_axis_points: Whether to display axis points markers. Default is True.
        axis_points_colors: Two colors for axis points markers (legacy parameter).
        axis_points_size: Marker size for axis points (legacy parameter).
        line_name: Trace name for axis points line (legacy parameter).
        line_color: Line color for axis points line (legacy parameter).
        line_width: Line width for axis points line (legacy parameter).
        line_dash: Dash style for axis points line (legacy parameter).
        config: Configuration object for styling. Overrides individual parameters.

    Raises:
        ValueError: If axis_points shape is not (2, 3).
    """
    # Validate inputs
    axis_points = np.asarray(axis_points)
    _validate_axis_points(axis_points)

    # Set defaults
    if axis_points_labels is None:
        axis_points_labels = ["pt0", "pt1"]

    # Build configuration from individual parameters if config not provided
    if config is None:
        kp_style = KeyPointsStyle(
            colors=axis_points_colors or ("green", "blue"),
            size=axis_points_size if axis_points_size is not None else 10.0,
        )
        line_style = LineStyle(
            name=line_name or "Key points line",
            color=line_color or "green",
            width=line_width if line_width is not None else 4.0,
            dash=line_dash or "dash",
        )
        config = AxisPointsLineConfig(
            axis_points_style=kp_style,
            line_style=line_style,
            show_axis_points=show_axis_points,
        )

    # Add traces
    if config.show_axis_points:
        _add_axis_points_trace(
            fig, axis_points, axis_points_labels, config.axis_points_style
        )
    _add_line_trace(fig, axis_points, config.line_style)


def add_span_points(
    fig: go.Figure,
    span_points: np.ndarray,
    *,
    span_name: str | None = None,
    span_color: str | None = None,
    span_marker_size: float | None = None,
    config: SpanStyle | None = None,
) -> None:
    """Add span points to a figure.

    Args:
        fig: Plotly figure to update.
        span_points: Array of shape (N, 3) containing span points along the line.
        span_name: Trace name for span points (legacy parameter).
        span_color: Marker color for span points (legacy parameter).
        span_marker_size: Marker size for span points (legacy parameter).
        config: Configuration object for styling. Overrides individual parameters.

    Raises:
        ValueError: If span_points is not an (N, 3) array.
    """
    # Validate inputs
    span_points = np.asarray(span_points)
    _validate_span_points(span_points)

    # Build configuration from individual parameters if config not provided
    if config is None:
        config = SpanStyle(
            name=span_name or "Span",
            color=span_color or "orange",
            marker_size=span_marker_size
            if span_marker_size is not None
            else 3.0,
        )

    # Add trace
    _add_span_trace(fig, span_points, config)


# def add_axis_points_line_and_span(
#     fig: go.Figure,
#     axis_points: np.ndarray,
#     span_points: np.ndarray,
#     *,
#     axis_points_labels: Iterable[str] | None = None,
#     axis_points_colors: tuple[str, str] | None = None,
#     axis_points_size: float | None = None,
#     line_name: str | None = None,
#     line_color: str | None = None,
#     line_width: float | None = None,
#     line_dash: str | None = None,
#     span_name: str | None = None,
#     span_color: str | None = None,
#     span_marker_size: float | None = None,
#     config: KeyPointsLineConfig | None = None,
# ) -> None:
#     """Add axis points, the line linking them, and the span to a figure.

#     This function adds three traces to the figure:
#     1. Axis points as markers with text labels
#     2. A line connecting the axis points
#     3. Span points as markers below the line

#     .. deprecated::
#         Use :func:`add_axis_points_line` and :func:`add_span_points` separately
#         for more flexibility.

#     Args:
#         fig: Plotly figure to update.
#         axis_points: Array of shape (2, 3) containing the axis points.
#         span_points: Array of shape (N, 3) containing span points along the line.
#         axis_points_labels: Optional labels for axis points. Defaults to ["pt0", "pt1"].
#         axis_points_colors: Two colors for axis points markers (legacy parameter).
#         axis_points_size: Marker size for axis points (legacy parameter).
#         line_name: Trace name for axis points line (legacy parameter).
#         line_color: Line color for axis points line (legacy parameter).
#         line_width: Line width for axis points line (legacy parameter).
#         line_dash: Dash style for axis points line (legacy parameter).
#         span_name: Trace name for span points (legacy parameter).
#         span_color: Marker color for span points (legacy parameter).
#         span_marker_size: Marker size for span points (legacy parameter).
#         config: Configuration object for styling. Overrides individual parameters.

#     Raises:
#         ValueError: If axis_points shape is not (2, 3) or span_points is not (N, 3).
#     """
#     # Validate inputs
#     axis_points = np.asarray(axis_points)
#     span_points = np.asarray(span_points)
#     _validate_axis_points(axis_points)
#     _validate_span_points(span_points)

#     # Set defaults
#     if axis_points_labels is None:
#         axis_points_labels = ["pt0", "pt1"]

#     # Build configuration from individual parameters if config not provided
#     if config is None:
#         kp_style = KeyPointsStyle(
#             colors=axis_points_colors or ("green", "blue"),
#             size=axis_points_size if axis_points_size is not None else 10.0,
#         )
#         line_style = LineStyle(
#             name=line_name or "Key points line",
#             color=line_color or "green",
#             width=line_width if line_width is not None else 4.0,
#             dash=line_dash or "dash",
#         )
#         span_style = SpanStyle(
#             name=span_name or "Span",
#             color=span_color or "orange",
#             marker_size=span_marker_size
#             if span_marker_size is not None
#             else 3.0,
#         )
#         config = KeyPointsLineConfig(
#             axis_points_style=kp_style,
#             line_style=line_style,
#             span_style=span_style,
#         )

#     # Add traces
#     _add_axis_points_trace(
#         fig, axis_points, axis_points_labels, config.axis_points_style
#     )
#     _add_line_trace(fig, axis_points, config.line_style)
#     _add_span_trace(fig, span_points, config.span_style)


# def add_obstacle_plane_and_intersection(
#     fig: go.Figure,
#     obstacle: np.ndarray,
#     axis_points: np.ndarray,
#     span_points: np.ndarray,
#     *,
#     plane_scale: float | None = None,
#     plane_grid_size: int | None = None,
#     fine_tuning: bool | None = None,
#     obstacle_name: str | None = None,
#     plane_name: str | None = None,
#     intersection_name: str | None = None,
#     obstacle_color: str | None = None,
#     obstacle_size: float | None = None,
#     plane_colorscale: str | None = None,
#     plane_opacity: float | None = None,
#     intersection_color: str | None = None,
#     intersection_size: float | None = None,
#     config: ObstaclePlaneConfig | None = None,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Add obstacle, orthogonal plane, and intersection points to a figure.

#     This function adds three components:
#     1. An obstacle point in 3D space
#     2. An orthogonal plane perpendicular to the axis_points line through the obstacle
#     3. Intersection points where the span curve crosses the plane

#     Args:
#         fig: Plotly figure to update.
#         obstacle: Point on the plane of shape (3,).
#         axis_points: Array of shape (2, 3) defining the line for plane normal.
#         span_points: Array of shape (N, 3) containing span points for intersection.
#         plane_scale: Extent of the plane mesh (legacy parameter).
#         plane_grid_size: Grid size for the plane mesh (legacy parameter).
#         fine_tuning: Use fine tuning when computing intersection (legacy parameter).
#         obstacle_name: Trace name for obstacle (legacy parameter).
#         plane_name: Trace name for plane (legacy parameter).
#         intersection_name: Trace name for intersection points (legacy parameter).
#         obstacle_color: Marker color for obstacle (legacy parameter).
#         obstacle_size: Marker size for obstacle (legacy parameter).
#         plane_colorscale: Plotly colorscale for plane (legacy parameter).
#         plane_opacity: Plane surface opacity (legacy parameter).
#         intersection_color: Marker color for intersection points (legacy parameter).
#         intersection_size: Marker size for intersection points (legacy parameter).
#         config: Configuration object for plane and styling. Overrides individual parameters.

#     Returns:
#         Tuple of:
#             - intersections: Array of intersection points between span and plane
#             - u_plane: First orthonormal basis vector for the plane
#             - v_plane: Second orthonormal basis vector for the plane

#     Raises:
#         ValueError: If input arrays have incorrect shapes.
#     """
#     # Validate inputs
#     obstacle = np.asarray(obstacle)
#     axis_points = np.asarray(axis_points)
#     span_points = np.asarray(span_points)
#     _validate_obstacle(obstacle)
#     _validate_axis_points(axis_points)
#     _validate_span_points(span_points)

#     # Build configuration from individual parameters if config not provided
#     if config is None:
#         obstacle_style = ObstacleStyle(
#             name=obstacle_name or "Obstacle",
#             color=obstacle_color or "red",
#             size=obstacle_size if obstacle_size is not None else 12.0,
#         )
#         plane_style = PlaneStyle(
#             name=plane_name or "Orthogonal Plane",
#             colorscale=plane_colorscale or "Greens",
#             opacity=plane_opacity if plane_opacity is not None else 0.2,
#         )
#         intersection_style = IntersectionStyle(
#             name=intersection_name or "Intersection",
#             color=intersection_color or "purple",
#             size=intersection_size if intersection_size is not None else 4.0,
#         )
#         config = ObstaclePlaneConfig(
#             plane_scale=plane_scale if plane_scale is not None else 10.0,
#             plane_grid_size=plane_grid_size
#             if plane_grid_size is not None
#             else 10,
#             fine_tuning=fine_tuning if fine_tuning is not None else False,
#             obstacle_style=obstacle_style,
#             plane_style=plane_style,
#             intersection_style=intersection_style,
#         )

#     # Compute plane geometry
#     plane_normal = compute_plane_normal(axis_points)
#     u_plane, v_plane, _ = plane_from_line(obstacle, plane_normal)
#     x_plane, y_plane, z_plane = meshgrid_plane(
#         u_plane,
#         v_plane,
#         obstacle,
#         scale_plane=config.plane_scale,
#         grid_size_plane=config.plane_grid_size,
#     )

#     # Add traces
#     _add_obstacle_trace(fig, obstacle, config.obstacle_style)
#     _add_plane_surface_trace(
#         fig, x_plane, y_plane, z_plane, config.plane_style
#     )

#     # Compute and add intersections
#     intersections = intersection_curve_plane(
#         plane_normal, obstacle, span_points, fine_tuning=config.fine_tuning
#     )
#     _add_intersection_trace(fig, intersections, config.intersection_style)

#     return intersections, u_plane, v_plane


# ============================================================================
# High-Level Plotting Functions
# ============================================================================


def plot_distance_engine(
    distance_engine: DistanceEngine,
    distance_result: DistanceResult | None = None,
    fig: go.Figure | None = None,
    title_addendum: str | None = None,
    show_axis_points: bool = True,
    show_curve: bool = True,
    show_plane: bool = True,
    show_distance_result: bool = True,
    show_projections: bool = True,
    plane_scale: float = 10.0,
    plane_grid_size: int = 3,
    axis_labels: Iterable[str] | None = None,
    axis_color: str = "darkviolet",
    curve_color: str = "darkblue",
    plane_color: str = "gold",
    projection_color: str = "firebrick",
) -> go.Figure:
    """Plot DistanceEngine components including axis, curve, plane, and distance result.

    This function creates a comprehensive 3D visualization of the DistanceEngine,
    showing the axis line, curve points, optional plane, and distance calculations.

    Args:
        distance_engine: The DistanceEngine instance to visualize.
        distance_result: Optional DistanceResult from plane_distance computation.
        fig: Existing figure to add to, or None to create a new figure.
        show_axis_points: Whether to show axis start/end point markers.
        show_curve: Whether to show the curve points.
        show_plane: Whether to show the distance plane (requires distance_result).
        show_distance_result: Whether to show distance result points and vectors.
        show_projections: Whether to show projection lines (requires distance_result).
        plane_scale: Scale of the plane visualization.
        plane_grid_size: Grid density for plane mesh.
        axis_labels: Labels for axis start and end points.
        axis_color: Color for axis line and points.
        curve_color: Color for curve points.
        plane_color: Color for plane surface.
        projection_color: Color for projection vectors.
        title_addendum: Optional string to append to legend groups for uniqueness.

    Returns:
        Plotly figure with all requested components.

    Raises:
        AttributeError: If distance_engine is missing required attributes.
        ValueError: If show_distance_result is True but distance_result is None.

    Example:
        >>> engine = DistanceEngine()
        >>> engine.add_span_frame(axis_start, axis_end)
        >>> engine.add_curves(curve_points)
        >>> result = engine.plane_distance(point_base)
        >>> fig = plot_distance_engine(engine, result, show_plane=True)
    """
    # Validate inputs
    if not hasattr(distance_engine, "axis_start") or not hasattr(
        distance_engine, "axis_end"
    ):
        raise AttributeError(
            "DistanceEngine must have axis_start and axis_end defined. "
            "Call add_span_frame() first."
        )

    if show_distance_result and distance_result is None:
        raise ValueError(
            "distance_result must be provided when show_distance_result=True"
        )

    # Create or reuse figure
    title_addendum = f" - {title_addendum}" if title_addendum else ""

    if fig is None:
        fig = go.Figure()
        fig.update_layout(
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="data",
            ),
            title="Distance Engine Visualization",
        )

    # 1. Plot axis line and points
    axis_points = np.array(
        [distance_engine.axis_start, distance_engine.axis_end]
    )
    if axis_labels is None:
        axis_labels = ["span origin", "span X axis"]

    add_axis_points_line(
        fig,
        axis_points,
        axis_points_labels=axis_labels,
        show_axis_points=show_axis_points,
        line_color=axis_color,
        line_width=4.0,
        axis_points_colors=(axis_color, axis_color),
    )

    # 2. Plot curve points
    if show_curve and hasattr(distance_engine, "curve_points"):
        add_span_points(
            fig,
            distance_engine.curve_points,
            span_name="Curve",
            span_color=curve_color,
            span_marker_size=4.0,
        )

    # 4. Plot plane if requested
    if show_plane:
        # Compute plane mesh
        u_plane = distance_engine.u_plane
        v_plane = distance_engine.v_plane
        point_on_plane = distance_engine.point_base

        if distance_result is not None:
            plane_scale = max(
                distance_result.distance_3d * 2, plane_scale
            )  # Scale plane based on distance for better visualization

        # Create plane grid
        x_plane, y_plane, z_plane = meshgrid_plane(
            u_plane,
            v_plane,
            point_on_plane,
            scale_plane=plane_scale,
            grid_size_plane=plane_grid_size,
        )

        # Add plane surface
        plot_surface(fig, title_addendum, plane_color, x_plane, y_plane, z_plane)

    # 3. Plot distance result components
    if show_distance_result and distance_result is not None:
        # Add the two key points
        fig.add_trace(
            go.Scatter3d(
                x=[distance_result.point_1[0], distance_result.point_2[0]],
                y=[distance_result.point_1[1], distance_result.point_2[1]],
                z=[distance_result.point_1[2], distance_result.point_2[2]],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color=["darkred", "darkred"],
                    symbol=["cross", "cross"],
                ),
                text=["Obstacle", "Intersection"],
                textposition="top center",
                name="Distance Points",
                legendgroup="distance" + title_addendum,
            )
        )

        # Add 3D distance line
        fig.add_trace(
            go.Scatter3d(
                x=[distance_result.point_1[0], distance_result.point_2[0]],
                y=[distance_result.point_1[1], distance_result.point_2[1]],
                z=[distance_result.point_1[2], distance_result.point_2[2]],
                mode="lines",
                line=dict(color="darkred", width=4, dash="solid"),
                name=f"Plane Distance: {distance_result.distance_3d:.3f}m",
                legendgroup="distance" + title_addendum,
            )
        )
        # Add distance text label
        midpoint = (distance_result.point_1 + distance_result.point_2) / 2
        fig.add_trace(
            go.Scatter3d(
                x=[midpoint[0]],
                y=[midpoint[1]],
                z=[midpoint[2]],
                mode="text",
                text=[f"{distance_result.distance_3d:.2f} m"],
                textposition="top center",
                textfont=dict(size=14, color="black"),
                name="Distance Text",
                legendgroup="distance" + title_addendum,
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Add projection vectors if requested
        if show_projections:
            # Get projection points
            u_proj, v_proj = distance_result.projection_points(
                distance_result.point_1
            )

            # U projection line
            fig.add_trace(
                go.Scatter3d(
                    x=[distance_result.point_1[0], u_proj[0]],
                    y=[distance_result.point_1[1], u_proj[1]],
                    z=[distance_result.point_1[2], u_proj[2]],
                    mode="lines+markers",
                    line=dict(color=projection_color, width=4, dash="dot"),
                    marker=dict(
                        symbol="diamond", size=4, color=projection_color
                    ),
                    name=f"U Projection: {distance_result.distance_projection_u:.2f}m",
                    hovertemplate=(
                        "%{fullData.name}<br>"
                        "X: %{x:.3f}<br>"
                        "Y: %{y:.3f}<br>"
                        "Z: %{z:.3f}<br>"
                        "<extra></extra>"
                    ),
                    legendgroup="projections" + title_addendum,
                )
            )

            # V projection line
            fig.add_trace(
                go.Scatter3d(
                    x=[distance_result.point_1[0], v_proj[0]],
                    y=[distance_result.point_1[1], v_proj[1]],
                    z=[distance_result.point_1[2], v_proj[2]],
                    mode="lines+markers",
                    line=dict(color=projection_color, width=4, dash="dot"),
                    marker=dict(
                        symbol="diamond", size=4, color=projection_color
                    ),
                    name=f"V Projection: {distance_result.distance_projection_v:.2f}m",
                    hovertemplate=(
                        "%{fullData.name}<br>"
                        "X: %{x:.3f}<br>"
                        "Y: %{y:.3f}<br>"
                        "Z: %{z:.3f}<br>"
                        "<extra></extra>"
                    ),
                    legendgroup="projections" + title_addendum,
                )
            )

    return fig

def plot_surface(fig, title_addendum, plane_color, x_plane, y_plane, z_plane):
    fig.add_trace(
            go.Surface(
                x=x_plane,
                y=y_plane,
                z=z_plane,
                opacity=0.2,
                colorscale=[[0, plane_color], [1, plane_color]],
                name="Distance Plane",
                showscale=False,
                legendgroup="plane" + title_addendum,
                showlegend=True,
            )
        )
