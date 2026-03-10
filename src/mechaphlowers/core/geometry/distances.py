# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from typing import Any, Literal

import numpy as np

from mechaphlowers.core.geometry.planes import (
    change_local_frame,
    intersection_curve_plane,
    plane_from_line,
)
from mechaphlowers.plotting.plot_distances import plot_distance_engine


def points_distance_inside_plane(
    point_base: np.ndarray,
    point_target: np.ndarray,
    u_plane: np.ndarray,
    v_plane: np.ndarray,
) -> tuple[float, float, float]:
    """Compute the distance between two points in 3D inside a plane.

    The plane is defined by its basis vectors u_plane and v_plane, which are orthogonal and normalized.
    The function compute 3D distance between the two points, as well as the projections of this distance onto the plane basis vectors.

    Args:
        point_base: The first point in 3D space (numpy array of shape (3,)).
        point_target: The second point in 3D space (numpy array of shape (3,)).
        u_plane: The first basis vector of the plane (numpy array of shape (3,)).
        v_plane: The second basis vector of the plane (numpy array of shape (3,)).

    Returns:
        A tuple containing the 3D distance and the projections onto the plane basis vectors (distance_3d, distance_projection_u, distance_projection_v).
    """

    # Calculate difference vector
    diff_vector = point_target - point_base
    print(
        f"\nDifference vector: ({diff_vector[0]:.4f}, {diff_vector[1]:.4f}, {diff_vector[2]:.4f})"
    )

    # Calculate 3D distance
    distance_3d = np.linalg.norm(diff_vector)

    # Project onto plane frame basis vectors
    # u_plane: first basis vector in the plane (y-direction in plane frame)
    # v_plane: second basis vector in the plane (z-direction in plane frame)
    distance_projection_u = np.dot(diff_vector, u_plane)
    distance_projection_v = np.dot(diff_vector, v_plane)

    # Verify: the projections should satisfy Pythagorean theorem
    projected_distance = np.sqrt(
        distance_projection_u**2 + distance_projection_v**2
    )
    if abs(projected_distance - distance_3d) > 1e-6:
        raise ValueError(
            f"Projected distance ({projected_distance:.6f} m) does not match 3D distance ({distance_3d:.6f} m) - check calculations!"
        )

    return distance_3d, distance_projection_u, distance_projection_v


def get_projection_points(
    origin_point: np.ndarray,
    projection_u: float,
    projection_v: float,
    u_plane: np.ndarray,
    v_plane: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the projection points on the plane basis vectors from an origin point and the projections.

    Args:
        origin_point: The original point in 3D space from which projections are calculated (numpy array of shape (3,)).
        projection_u: The scalar projection distance along the u_plane basis vector.
        projection_v: The scalar projection distance along the v_plane basis vector.
        u_plane: The first basis vector of the plane (numpy array of shape (3,)).
        v_plane: The second basis vector of the plane (numpy array of shape (3,))

    Returns:
        A tuple containing the projection points on the plane basis vectors (u_projection_point, v_projection_point).
    """
    u_projection_point = origin_point + projection_u * u_plane
    v_projection_point = origin_point + projection_v * v_plane
    return u_projection_point, v_projection_point


class DistanceResult:
    def __init__(
        self,
        point_base: np.ndarray,
        point_target: np.ndarray,
        u_plane: np.ndarray,
        v_plane: np.ndarray,
        distance_3d: float,
        distance_projection_u: float,
        distance_projection_v: float,
    ):
        self.point_base = point_base
        self.point_target = point_target
        self.distance_3d = distance_3d
        self.distance_projection_u = distance_projection_u
        self.distance_projection_v = distance_projection_v
        self.u_plane = u_plane
        self.v_plane = v_plane

    def __str__(self):
        return (
            f"DistanceResult:\n"
            f"  Point Base: ({self.point_base[0]:.4f}, {self.point_base[1]:.4f}, {self.point_base[2]:.4f})\n"
            f"  Point Target: ({self.point_target[0]:.4f}, {self.point_target[1]:.4f}, {self.point_target[2]:.4f})\n"
            f"  Distance 3D: {self.distance_3d:.6f} m\n"
            f"  Distance Projection U: {self.distance_projection_u:.6f} m\n"
            f"  Distance Projection V: {self.distance_projection_v:.6f} m\n"
            f"  U Plane Vector: ({self.u_plane[0]:.4f}, {self.u_plane[1]:.4f}, {self.u_plane[2]:.4f})\n"
            f"  V Plane Vector: ({self.v_plane[0]:.4f}, {self.v_plane[1]:.4f}, {self.v_plane[2]:.4f})\n"
        )

    def __repr__(self):
        return "<DistanceResult>\n" + self.__str__()

    def projection_points(
        self, origin_point: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            origin_point + self.distance_projection_u * self.u_plane,
            origin_point + self.distance_projection_v * self.v_plane,
        )


class DistanceEngine:
    """DistanceEngine distance computation between a point and a curve in 3D space.

    It uses a defined plane for the distance calculation. The plane is defined by a span frame, which is determined by two points (start and end of the span). The engine allows to add curves and span frames, and then compute the distance from a given point to the curve along the plane defined by the span frame. The result is returned as a DistanceResult object containing the distance information and projection details.

    Example:
        >>> de = DistanceEngine()
        >>> de.add_curves(curve_points)
        >>> de.add_span_frame(span_start, span_end)
        >>> distance_result = de.plane_distance(obstacle_point)
        >>> fig = de.plot(distance_result)
        >>> fig.show()
    """

    def __init__(self):
        pass

    def add_curves(self, curve_points: np.ndarray):
        """add curves to the engine, which will be used for distance calculations. The curves are defined by their points in 3D space.

        Args:
            curve_points: A numpy array of shape (N, 3) representing the points of one or more curves in 3D space, concatenated along the first dimension.
        """
        self.curve_points = curve_points

    def add_span_frame(self, x_axis_start: np.ndarray, x_axis_end: np.ndarray):
        """Add a span frame to the engine, which will be used for distance calculations.

        The span frame is defined by its X axis start and end points in 3D space.
        This frame will be used to define the plane for distance calculations, with the normal vector of the plane being vertical (Z direction) and the other two basis vectors defined in the XY plane along the span direction and perpendicular to it.

        Warning - Z is always oriented upwards, which is important for the distance calculations.

        Args:
            x_axis_start: A numpy array of shape (3,) representing the start point of the span frame.
            x_axis_end: A numpy array of shape (3,) representing the end point of the span frame.
        """
        self.axis_start = x_axis_start.copy()
        self.axis_end = x_axis_end.copy()

        self.axis_start[2] = 0  # Ensure Z=0 for span frame
        self.axis_end[2] = 0

        self.line_direction = self.axis_end - self.axis_start
        self.line_direction[2] = 0  # ensure projection onto XY plane
        self.line_direction_normalized = self.line_direction / np.linalg.norm(
            self.line_direction
        )

    @property
    def axis_points(self) -> np.ndarray:
        """Return the axis points as a numpy array of shape (2, 3) containing the start and end points of the span frame."""
        return np.array([self.axis_start, self.axis_end])

    def define_distance_plane(
        self, point: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Define the distance plane based on a given point.

        Args:
            point: A numpy array of shape (3,) representing the point from which the distance plane belongs. This point will be used as the origin of the plane, and the plane will be defined with a normal vector vertical to the span frame (Z direction) and two basis vectors in the XY plane.

        Returns:
            A tuple containing the basis vectors of the plane (u_plane, v_plane).
        """
        # Define plane normal (vertical)
        self.u_plane, self.v_plane, _ = plane_from_line(
            point=point, plane_normal=self.line_direction_normalized
        )
        return self.u_plane, self.v_plane

    def plane_distance(
        self,
        point_base: np.ndarray,
        frame: Literal["span", "section"] = "span",
    ) -> DistanceResult:
        if frame == "span":
            # Transform the base point from the span frame to the local frame defined by the axis.
            self.point_base = change_local_frame(
                local_frame_origin=self.axis_start,
                local_frame_x_axis=self.axis_end,
                local_point=point_base,
            )
        elif frame == "section":
            # In section frame, the provided point is already expressed in the correct coordinates.
            self.point_base = point_base
        else:
            raise ValueError(
                f"Unsupported frame '{frame}'. Expected 'span' or 'section'."
            )
        self.define_distance_plane(point=self.point_base)

        intersection_point = intersection_curve_plane(
            curve_points=self.curve_points,
            point_on_plane=self.point_base,
            plane_normal=self.line_direction_normalized,
            fine_tuning=True,
        )

        if intersection_point is None:
            raise ValueError(
                "No intersection found between curve and plane - check inputs!"
            )
        if len(intersection_point) > 1:
            raise ValueError(
                "Multiple intersections found - expected only one! Check curve and plane configuration."
            )

        distance_3d, distance_projection_u, distance_projection_v = (
            points_distance_inside_plane(
                point_base=self.point_base,
                point_target=intersection_point[0],
                u_plane=self.u_plane,
                v_plane=self.v_plane,
            )
        )

        return DistanceResult(
            point_base=self.point_base,
            point_target=intersection_point[0],
            u_plane=self.u_plane,
            v_plane=self.v_plane,
            distance_3d=distance_3d,
            distance_projection_u=distance_projection_u,
            distance_projection_v=distance_projection_v,
        )

    # this method is a viewer for user and is intended to be moved after refactor of the plotting module
    def plot(
        self,
        distance_result: DistanceResult,
        show_plane: bool = True,
        show_projections: bool = True,
        **kwargs: Any,
    ):  # no typing for return here to avoid import plotly
        """Helper method to plot the distance result using the plot_distance_engine function from the plotting module.

        Args:
            distance_result: The DistanceResult object containing the distance information.
            show_plane: Boolean flag to indicate whether to show the distance plane.
            show_projections: Boolean flag to indicate whether to show the distance projection.
            **kwargs: Additional keyword arguments to pass to the plotting function.

        Returns:
            A plotly figure object containing the distance visualization.
        """
        return plot_distance_engine(
            self,
            distance_result,
            show_plane=show_plane,
            show_projections=show_projections,
            **kwargs,
        )
