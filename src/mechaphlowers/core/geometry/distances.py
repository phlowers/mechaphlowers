# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from typing import Literal

import numpy as np

from mechaphlowers.core.geometry.planes import (
    change_local_frame,
    intersection_curve_plane,
    plane_from_line,
)


def points_distance_inside_plane(
    point_base, point_target, u_plane, v_plane
) -> tuple[float, float, float]:
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
    origin_point, projection_u, projection_v, u_plane, v_plane
):
    u_projection_point = origin_point + projection_u * u_plane
    v_projection_point = origin_point + projection_v * v_plane
    return u_projection_point, v_projection_point


class DistanceResult:
    def __init__(
        self,
        point_base,
        point_target,
        u_plane,
        v_plane,
        distance_3d,
        distance_projection_u,
        distance_projection_v,
    ):
        self.point_1 = point_base
        self.point_2 = point_target
        self.distance_3d = distance_3d
        self.distance_projection_u = distance_projection_u
        self.distance_projection_v = distance_projection_v
        self.u_plane = u_plane
        self.v_plane = v_plane

    def __str__(self):
        return (
            f"DistanceResult:\n"
            f"  Point 1: ({self.point_1[0]:.4f}, {self.point_1[1]:.4f}, {self.point_1[2]:.4f})\n"
            f"  Point 2: ({self.point_2[0]:.4f}, {self.point_2[1]:.4f}, {self.point_2[2]:.4f})\n"
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
    def __init__(self):
        pass

    def add_curves(self, curve_points):
        self.curve_points = curve_points

    def add_span_frame(self, axis_start, axis_end):
        self.axis_start = axis_start.copy()
        self.axis_end = axis_end.copy()

        self.axis_start[2] = 0  # Ensure Z=0 for span frame
        self.axis_end[2] = 0

        self.line_direction = self.axis_end - self.axis_start
        self.line_direction[2] = 0  # ensure projection onto XY plane
        self.line_direction_normalized = self.line_direction / np.linalg.norm(
            self.line_direction
        )

    @property
    def axis_points(self):
        return np.array([self.axis_start, self.axis_end])

    def define_distance_plane(self, point):
        # Define plane normal (vertical)
        self.u_plane, self.v_plane, _ = plane_from_line(
            point=point, plane_normal=self.line_direction_normalized
        )
        return self.u_plane, self.v_plane

    def plane_distance(
        self, point_base, frame: Literal["span", "section"] = "span"
    ):
        if frame == "span":
            self.point_base = change_local_frame(
                local_frame_origin=self.axis_start,
                local_frame_x_axis=self.axis_end,
                local_point=point_base,
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

    def plot(self, distance_result: DistanceResult, **kwargs):
        from mechaphlowers.plotting.plot_distances import plot_distance_engine

        return plot_distance_engine(self, distance_result, **kwargs)
