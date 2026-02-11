# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import logging
import numpy as np

logger = logging.getLogger(__name__)


def parametric_line_from_2_points(p1, p2):
    """Returns the coefficients of the line defined by two points p1 and p2 in 3D space.

    The line is represented in parametric form as:
    L(t) = p1 + t * line_direction_normalized

    where t is a scalar parameter and line_direction_normalized is the normalized direction vector of the line.
    """
    # Direction vector of the line between p1 and p2
    line_direction = p2 - p1
    line_direction_normalized = line_direction / np.linalg.norm(line_direction)

    return p1, line_direction, line_direction_normalized


def line_function_from_2_points(p1, p2):
    """Returns a function that represents the line defined by two points p1 and p2 in 3D space.

    The returned function takes a scalar parameter t and returns the point on the line corresponding to that parameter.
    """
    p1, line_direction, _ = parametric_line_from_2_points(p1, p2)

    def line_function(t):
        return p1 + t * line_direction

    return line_function


def plane_from_line(point, plane_normal):
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
    u_plane, v_plane, point, scale_plane=150, grid_size_plane=20
):
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
    plane_normal, point_on_plane, curve_points, fine_tuning=False
):
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

    intersections = []

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
                    intersections.append(intersection_point)
        else:
            raise ValueError(
                f"Points are on the same side of the plane - no intersection!"
            )

        if intersections:
            intersections = np.array(intersections)
        else:
            logger.warning("No intersections found between span and plane!")
            intersections = np.array([])
    else:
        # Just return the 2 closest points for visualization, without checking if they are on opposite sides
        intersections = curve_points[closest_indices]

    return intersections
