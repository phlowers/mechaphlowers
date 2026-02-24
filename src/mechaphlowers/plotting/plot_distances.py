"""Plotting helpers for distance and plane visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, NamedTuple

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]

from mechaphlowers.core.geometry.planes import (
    meshgrid_plane,
)
from mechaphlowers.plotting.plot_config import (
    distance_text,
    line3d_trace,
    projected_distance_trace,
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


def add_surface(fig, title_addendum, plane_color, x_plane, y_plane, z_plane):
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


def plot_points(
    fig,
    points,
    text: List[str],
    name: str,
    marker: dict | None = None,
    text_position: str = "top center",
    legend_group: str | None = None,
):
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers+text",
            marker=marker,
            text=text,
            textposition=text_position,
            name=name,
            legendgroup=legend_group,
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


def plot_distance_points(
    fig, distance_points, color, symbol, text, title_addendum
):
    plot_points(
        fig,
        distance_points,
        marker=dict(
            size=10,
            color=color,
            symbol=symbol,
        ),
        name="Distance Points",
        text=text,
        text_position="top center",
        legend_group="distance" + title_addendum,
    )


def plot_3d_line(
    fig: go.Figure,
    point_1: tuple,
    point_2: tuple,
    distance_float: float,
    title_addendum: str,
):
    fig.add_trace(
        go.Scatter3d(
            x=[point_1[0], point_2[0]],
            y=[point_1[1], point_2[1]],
            z=[point_1[2], point_2[2]],
            mode=line3d_trace.scatter_mode,
            line=line3d_trace.line,
            name=f"{line3d_trace.name}: {distance_float:.3f}m",
            legendgroup=line3d_trace.legend_group + " " + title_addendum,
        )
    )


def plot_text(
    fig: go.Figure,
    text: str,
    title_addendum: str,
    position,
) -> None:
    """Add text label to figure.

    Args:
    text: DistanceResult containing distance_3d attribute for label text.
    fig: Plotly figure to add text to.
    title_addendum: String to append to legend group for uniqueness.
    position: 3D coordinates for text label placement.
    trace_config: TraceConfig object for customizing the trace appearance.

    """

    fig.add_trace(
        go.Scatter3d(
            x=[position[0]],
            y=[position[1]],
            z=[position[2]],
            mode=distance_text.scatter_mode,
            text=[distance_text.text_format(text)],
            textposition=distance_text.text_position,
            textfont=distance_text.text_font,
            name=distance_text.name,
            legendgroup=distance_text.legend_group + title_addendum,
            showlegend=distance_text.show_legend,
            hoverinfo=distance_text.hoverinfo,
        )
    )


def plot_projected_distances(
    fig: go.Figure,
    position: tuple,
    distance: float,
    name: str,
    title_addendum: str,
    projection_color: str,
    vect_proj: tuple,
):
    fig.add_trace(
        go.Scatter3d(
            x=[position[0], vect_proj[0]],
            y=[position[1], vect_proj[1]],
            z=[position[2], vect_proj[2]],
            mode=projected_distance_trace.scatter_mode,
            line=dict(
                color=projection_color,
                width=projected_distance_trace.width,
                dash=projected_distance_trace.dash,
            ),
            marker=dict(
                symbol=projected_distance_trace.marker_symbol,
                size=projected_distance_trace.size,
                color=projection_color,
            ),
            name=f"{name}: {distance:.2f}m",
            hovertemplate=(
                "%{fullData.name}<br>"
                "X: %{x:.3f}<br>"
                "Y: %{y:.3f}<br>"
                "Z: %{z:.3f}<br>"
                "<extra></extra>"
            ),
            legendgroup=projected_distance_trace.legend_group + title_addendum,
        )
    )


# ============================================================================
# High-Level Plotting Functions
# ============================================================================


def plot_distance_engine(
    distance_engine: DistanceEngine,
    distance_result: DistanceResult | None = None,
    fig: go.Figure | None = None,
    title_addendum: str | None = None,
    plane_scale: float = 10.0,
    plane_grid_size: int = 3,
    axis_labels: Iterable[str] | None = None,
    axis_color: str = "darkviolet",
    curve_color: str = "darkblue",
    plane_color: str = "gold",
    projection_color: str = "firebrick",
    force_layout: bool = True,
    show_axis_points: bool = True,
    show_curve: bool = True,
    show_plane: bool = True,
    show_distance_result: bool = True,
    show_projections: bool = True,
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

    Examples:
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

    if fig is None or force_layout:
        fig.update_layout(
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="data",
            ),
            title="Distance Analysis Engine Visualization",
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
        add_surface(
            fig, title_addendum, plane_color, x_plane, y_plane, z_plane
        )

    # 3. Plot distance result components
    if show_distance_result and distance_result is not None:
        # Add the two key points
        plot_distance_points(
            fig=fig,
            distance_points=np.array(
                [distance_result.point_base, distance_result.point_target]
            ),
            color=["darkred", "darkred"],
            symbol=["cross", "cross"],
            text=["Obstacle", "Intersection"],
            title_addendum=title_addendum,
        )

        # Add 3D distance line
        plot_3d_line(
            fig,
            distance_result.point_base,
            distance_result.point_target,
            distance_result.distance_3d,
            title_addendum,
        )
        # Add distance text label
        midpoint = (
            distance_result.point_base + distance_result.point_target
        ) / 2
        plot_text(fig, distance_result.distance_3d, title_addendum, midpoint)

        # Add projection vectors if requested
        if show_projections:
            # Get projection points
            u_proj, v_proj = distance_result.projection_points(
                distance_result.point_base
            )

            # U,V projection line
            plot_projected_distances(
                fig,
                distance_result.point_base,
                distance_result.distance_projection_u,
                "U Projection",
                title_addendum,
                projection_color,
                u_proj,
            )
            plot_projected_distances(
                fig,
                distance_result.point_base,
                distance_result.distance_projection_v,
                "V Projection",
                title_addendum,
                projection_color,
                v_proj,
            )

    return fig
