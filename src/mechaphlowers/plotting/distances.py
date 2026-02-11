"""Plotting helpers for distance and plane visualization."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]

from mechaphlowers.core.geometry.planes import (
	intersection_curve_plane,
	meshgrid_plane,
	plane_from_line,
)


def add_key_points_line_and_span(
	fig: go.Figure,
	key_points: np.ndarray,
	span_points: np.ndarray,
	*,
	key_points_labels: Iterable[str] | None = None,
	key_points_colors: Tuple[str, str] = ("green", "blue"),
	key_points_size: float = 10.0,
	line_name: str = "Key points line",
	line_color: str = "green",
	line_width: float = 4.0,
	line_dash: str = "dash",
	span_name: str = "Span",
	span_color: str = "orange",
	span_marker_size: float = 3.0,
) -> None:
	"""Add key points, the line linking them, and the span to a figure.

	Args:
		fig: Plotly figure to update.
		key_points: Array of shape (2, 3) containing the key points.
		span_points: Array of shape (N, 3) containing span points.
		key_points_labels: Optional labels for key points.
		key_points_colors: Two colors for key points markers.
		key_points_size: Marker size for key points.
		line_name: Trace name for key points line.
		line_color: Line color for key points line.
		line_width: Line width for key points line.
		line_dash: Dash style for key points line.
		span_name: Trace name for span points.
		span_color: Marker color for span points.
		span_marker_size: Marker size for span points.
	"""

	key_points = np.asarray(key_points)
	span_points = np.asarray(span_points)

	if key_points.shape != (2, 3):
		raise ValueError(
			"key_points must be an array of shape (2, 3) for the two key points"
		)
	if span_points.ndim != 2 or span_points.shape[1] != 3:
		raise ValueError("span_points must be an array of shape (N, 3)")

	if key_points_labels is None:
		key_points_labels = ["pt0", "pt1"]

	fig.add_trace(
		go.Scatter3d(
			x=key_points[:, 0],
			y=key_points[:, 1],
			z=key_points[:, 2],
			mode="markers+text",
			marker=dict(size=key_points_size, color=list(key_points_colors)),
			text=list(key_points_labels),
			textposition="top center",
			name="Key points",
			legendgroup="points",
		)
	)

	fig.add_trace(
		go.Scatter3d(
			x=key_points[:, 0],
			y=key_points[:, 1],
			z=key_points[:, 2],
			mode="lines",
			line=dict(color=line_color, width=line_width, dash=line_dash),
			name=line_name,
			legendgroup="line",
		)
	)

	fig.add_trace(
		go.Scatter3d(
			x=span_points[:, 0],
			y=span_points[:, 1],
			z=span_points[:, 2],
			mode="markers",
			marker=dict(color=span_color, size=span_marker_size),
			name=span_name,
			legendgroup="span",
		)
	)


def add_obstacle_plane_and_intersection(
	fig: go.Figure,
	obstacle: np.ndarray,
	key_points: np.ndarray,
	span_points: np.ndarray,
	*,
	plane_scale: float = 10.0,
	plane_grid_size: int = 10,
	fine_tuning: bool = False,
	obstacle_name: str = "Obstacle",
	plane_name: str = "Orthogonal Plane",
	intersection_name: str = "Intersection",
	obstacle_color: str = "red",
	obstacle_size: float = 12.0,
	plane_colorscale: str = "Greens",
	plane_opacity: float = 0.2,
	intersection_color: str = "purple",
	intersection_size: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Add obstacle, orthogonal plane, and intersection points to a figure.

	Args:
		fig: Plotly figure to update.
		obstacle: Point on the plane of shape (3,).
		key_points: Array of shape (2, 3) defining the line for plane normal.
		span_points: Array of shape (N, 3) containing span points.
		plane_scale: Extent of the plane mesh.
		plane_grid_size: Grid size for the plane mesh.
		fine_tuning: Use fine tuning when computing intersection.
		obstacle_name: Trace name for obstacle.
		plane_name: Trace name for plane.
		intersection_name: Trace name for intersection points.
		obstacle_color: Marker color for obstacle.
		obstacle_size: Marker size for obstacle.
		plane_colorscale: Plotly colorscale for plane.
		plane_opacity: Plane surface opacity.
		intersection_color: Marker color for intersection points.
		intersection_size: Marker size for intersection points.

	Returns:
		Tuple of (intersections, u_plane, v_plane).
	"""

	obstacle = np.asarray(obstacle)
	key_points = np.asarray(key_points)
	span_points = np.asarray(span_points)

	if obstacle.shape != (3,):
		raise ValueError("obstacle must be a 1D array of shape (3,)")
	if key_points.shape != (2, 3):
		raise ValueError(
			"key_points must be an array of shape (2, 3) for the two key points"
		)
	if span_points.ndim != 2 or span_points.shape[1] != 3:
		raise ValueError("span_points must be an array of shape (N, 3)")

	line_direction = key_points[1] - key_points[0]
	plane_normal = line_direction

	u_plane, v_plane, _ = plane_from_line(obstacle, plane_normal)
	x_plane, y_plane, z_plane = meshgrid_plane(
		u_plane,
		v_plane,
		obstacle,
		scale_plane=plane_scale,
		grid_size_plane=plane_grid_size,
	)

	fig.add_trace(
		go.Scatter3d(
			x=[obstacle[0]],
			y=[obstacle[1]],
			z=[obstacle[2]],
			mode="markers+text",
			marker=dict(size=obstacle_size, color=obstacle_color, symbol="cross"),
			text=[obstacle_name],
			textposition="top center",
			name=obstacle_name,
			legendgroup="obstacle",
		)
	)

	fig.add_trace(
		go.Surface(
			x=x_plane,
			y=y_plane,
			z=z_plane,
			opacity=plane_opacity,
			colorscale=plane_colorscale,
			name=plane_name,
			showscale=False,
		)
	)

	intersections = intersection_curve_plane(
		plane_normal, obstacle, span_points, fine_tuning=fine_tuning
	)

	if intersections.size > 0:
		fig.add_trace(
			go.Scatter3d(
				x=intersections[:, 0],
				y=intersections[:, 1],
				z=intersections[:, 2],
				mode="markers+text",
				marker=dict(size=intersection_size, color=intersection_color),
				text=[intersection_name] * intersections.shape[0],
				textposition="top center",
				name=intersection_name,
				legendgroup="intersection",
			)
		)

	return intersections, u_plane, v_plane
