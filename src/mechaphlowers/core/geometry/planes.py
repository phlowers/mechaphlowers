# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)


def change_local_frame(
    local_frame_origin: np.ndarray,
    local_frame_x_axis: np.ndarray,
    local_point: np.ndarray,
) -> np.ndarray:
    """change_local_frame transforms local coordinates defined by an origin and a span direction into absolute coordinates in the global frame.

    The local frame is defined such that:
    <ul>
        <li> The origin is the reference point in the global frame. </li>
        <li> The x-axis is aligned with the span direction projected onto the XY plane. </li>
        <li> The y-axis is perpendicular to the x-axis in the XY plane counterclockwise. </li>
        <li> The z-axis is vertical (same as global Z). </li>
    </ul>

    Args:
        local_frame_origin: Starting point of the span in global coordinates (3D).
        local_frame_x_axis: Ending point of the span in global coordinates (3D).
        local_point: Local coordinates of the point to transform (3D).
    Returns:
        Absolute coordinates of the point in the global frame (3D).
    """
    local_point = np.asarray(local_point)
    if local_point.shape != (3,):
        raise ValueError("local_point must be a 1D array of shape (3,)")

    local_frame_origin = np.asarray(local_frame_origin)
    local_frame_x_axis = np.asarray(local_frame_x_axis)

    # Compute span direction in XY plane
    delta_xy = local_frame_x_axis[:2] - local_frame_origin[:2]
    delta_norm = np.linalg.norm(delta_xy)

    if delta_norm == 0:
        raise ValueError("Span direction is zero in XY plane")

    # Construct orthonormal basis for the local frame
    # axis_x: unit vector along span in XY plane
    axis_x = delta_xy / delta_norm
    # axis_y: perpendicular to axis_x in XY plane (rotated 90° counterclockwise)
    axis_y = np.array([-axis_x[1], axis_x[0]])

    # Transform: absolute = origin + x_local * axis_x + y_local * axis_y + z_local * axis_z
    abs_xy = (
        local_frame_origin[:2]
        + local_point[0] * axis_x
        + local_point[1] * axis_y
    )
    abs_z = local_frame_origin[2] + local_point[2]

    return np.array([abs_xy[0], abs_xy[1], abs_z])


def parametric_line_from_2_points(
    p1: np.ndarray, p2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns the coefficients of the line defined by two points p1 and p2 in 3D space.

    The line is represented in parametric form as:
    L(t) = p1 + t * line_direction_normalized

    where t is a scalar parameter and line_direction_normalized is the normalized direction vector of the line.
    """
    # Direction vector of the line between p1 and p2
    line_direction = p2 - p1
    line_direction_normalized = line_direction / np.linalg.norm(line_direction)

    return p1, line_direction, line_direction_normalized


def line_function_from_2_points(
    p1: np.ndarray, p2: np.ndarray
) -> Callable[[float], np.ndarray]:
    """Returns a function that represents the line defined by two points p1 and p2 in 3D space.

    The returned function takes a scalar parameter t and returns the point on the line corresponding to that parameter.
    """
    p1, line_direction, _ = parametric_line_from_2_points(p1, p2)

    def line_function(t: float) -> np.ndarray:
        return p1 + t * line_direction

    return line_function


def compute_plane_normal(key_points: np.ndarray) -> np.ndarray:
    """Compute plane normal from two key points.

    Args:
        key_points: Array of shape (2, 3) defining start and end of line.

    Returns:
        Direction vector from first to second point.
    """
    return key_points[1] - key_points[0]


def plane_from_line(
    point: np.ndarray, plane_normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # First, find two orthogonal vectors in the plane
    if abs(plane_normal[0]) < 1e-6 and abs(plane_normal[1]) < 1e-6:
        u_plane = np.array([1.0, 0.0, 0.0])
    else:
        u_plane = np.array([-plane_normal[1], plane_normal[0], 0.0])
    u_plane = u_plane / np.linalg.norm(u_plane)

    v_plane = np.cross(plane_normal, u_plane)
    v_plane = v_plane / np.linalg.norm(v_plane)

    return u_plane, v_plane, point


def meshgrid_plane(
    u_plane: np.ndarray,
    v_plane: np.ndarray,
    point: np.ndarray,
    scale_plane: float = 150,
    grid_size_plane: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Create a grid on the plane

    s_plane = np.linspace(-scale_plane, scale_plane, grid_size_plane)
    t_plane_grid = np.linspace(-scale_plane, scale_plane, grid_size_plane)
    S_plane, T_plane = np.meshgrid(s_plane, t_plane_grid)

    # Coordinates of the plane
    X_plane_pts = point[0] + S_plane * u_plane[0] + T_plane * v_plane[0]
    Y_plane_pts = point[1] + S_plane * u_plane[1] + T_plane * v_plane[1]
    Z_plane_pts = point[2] + S_plane * u_plane[2] + T_plane * v_plane[2]

    return X_plane_pts, Y_plane_pts, Z_plane_pts


def intersection_curve_plane(
    plane_normal: np.ndarray,
    point_on_plane: np.ndarray,
    curve_points: np.ndarray,
    fine_tuning: bool = False,
) -> np.ndarray:
    # Find intersection between span1 and the orthogonal plane
    # REFACTORED: First find 2 closest points, then compute intersection

    plane_d_value = np.dot(plane_normal, point_on_plane)

    # Step 1: Calculate distance from each point to the plane
    # Distance = |dot(normal, point) - d| / ||normal||
    normal_magnitude = np.linalg.norm(plane_normal)
    # distances = np.abs(np.array([np.dot(plane_normal, pt) - plane_d_value for pt in spans1])) / normal_magnitude
    distances = (
        np.abs(curve_points @ plane_normal - plane_d_value) / normal_magnitude
    )

    # Step 2: Find the 2 closest points to the plane
    closest_indices = np.argsort(distances)[:2]
    closest_indices = np.sort(
        closest_indices
    )  # Sort to get them in order along the span

    # Step 3: Check if these 2 points are on opposite sides of the plane
    p1_signed_dist = (
        np.dot(plane_normal, curve_points[closest_indices[0]]) - plane_d_value
    ) / normal_magnitude
    p2_signed_dist = (
        np.dot(plane_normal, curve_points[closest_indices[1]]) - plane_d_value
    ) / normal_magnitude

    intersections = np.empty((0, 3), dtype=curve_points.dtype)

    if fine_tuning:
        # Only check if the segment crosses the plane (opposite signs)
        if p1_signed_dist * p2_signed_dist < 0:
            # Step 4: Calculate intersection using parametric equation
            p_start = curve_points[closest_indices[0]]
            p_end = curve_points[closest_indices[1]]
            segment_dir = p_end - p_start

            dot_normal_segment = np.dot(plane_normal, segment_dir)

            if abs(dot_normal_segment) > 1e-10:
                t = (
                    plane_d_value - np.dot(plane_normal, p_start)
                ) / dot_normal_segment

                if 0 <= t <= 1:
                    intersection_point = p_start + t * segment_dir
                    intersections = np.array([intersection_point])
        else:
            raise ValueError(
                "Points are on the same side of the plane - no intersection!"
            )

        if intersections.size == 0:
            logger.warning("No intersections found between span and plane!")
    else:
        # Just return the 2 closest points for visualization, without checking if they are on opposite sides
        intersections = curve_points[closest_indices]

    return intersections
