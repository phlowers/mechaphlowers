# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from mechaphlowers.core.geometry.planes import (
    change_local_frame,
    intersection_curve_plane,
    plane_from_line,
)
from mechaphlowers.core.geometry.points import rotate_vector


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
        self.signed_distance_projection_u = distance_projection_u
        self.signed_distance_projection_v = distance_projection_v
        self.u_plane = u_plane
        self.v_plane = v_plane

    @property
    def distance_projection_u(self) -> float:
        """Return the signed distance projection along the u_plane basis vector."""
        return abs(self.signed_distance_projection_u)

    @property
    def distance_projection_v(self) -> float:
        """Return the signed distance projection along the v_plane basis vector."""
        return abs(self.signed_distance_projection_v)

    def __str__(self):
        return (
            f"DistanceResult:\n"
            f"  Point Base: ({self.point_base[0]:.4f}, {self.point_base[1]:.4f}, {self.point_base[2]:.4f})\n"
            f"  Point Target: ({self.point_target[0]:.4f}, {self.point_target[1]:.4f}, {self.point_target[2]:.4f})\n"
            f"  Distance 3D: {self.distance_3d:.6f} m\n"
            f"  Distance Projection U: {self.signed_distance_projection_u:.6f} m\n"
            f"  Distance Projection V: {self.signed_distance_projection_v:.6f} m\n"
            f"  U Plane Vector: ({self.u_plane[0]:.4f}, {self.u_plane[1]:.4f}, {self.u_plane[2]:.4f})\n"
            f"  V Plane Vector: ({self.v_plane[0]:.4f}, {self.v_plane[1]:.4f}, {self.v_plane[2]:.4f})\n"
        )

    def __repr__(self):
        return "<DistanceResult>\n" + self.__str__()

    def projection_points(
        self, origin_point: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return get_projection_points(
            origin_point=origin_point,
            projection_u=self.signed_distance_projection_u,
            projection_v=self.signed_distance_projection_v,
            u_plane=self.u_plane,
            v_plane=self.v_plane,
        )

    def compute_new_frame(
        self, translation_vector: np.ndarray, angle_to_project: np.float64
    ) -> DistanceResult:
        point_base = self.point_base + translation_vector
        self.point_base = rotate_vector(point_base, angle_to_project)

        point_target = self.point_target + translation_vector
        self.point_target = rotate_vector(point_target, angle_to_project)

        self.u_plane = rotate_vector(self.u_plane, angle_to_project)
        self.v_plane = rotate_vector(self.v_plane, angle_to_project)

        return self

    def generate_with_reversed_y_axis(self):
        new_distance_result = deepcopy(self)
        new_distance_result.point_base[1] = -new_distance_result.point_base[1]
        new_distance_result.point_target[
            1
        ] = -new_distance_result.point_target[1]
        new_distance_result.u_plane[1] = -new_distance_result.u_plane[1]
        new_distance_result.v_plane[1] = -new_distance_result.v_plane[1]
        return new_distance_result


def minimize_bounded(
    func: Callable[[float], float],
    bounds: tuple[float, float],
    xtol: float = 1e-5,
    maxiter: int = 100,
) -> float:
    """Return the argmin of a scalar function on a bounded interval.

    Uses ``scipy.optimize.minimize_scalar(method="bounded")`` when scipy is
    installed, otherwise a pure-Python golden-section search.

    Args:
        func: Scalar function to minimize.
        bounds: Search interval ``(a, b)``.
        xtol: Absolute tolerance on the argmin.
        maxiter: Maximum number of iterations.

    Returns:
        The abscissa of the minimum found inside `bounds`.
    """
    a, b = bounds
    try:
        from scipy.optimize import minimize_scalar  # type: ignore
    except ImportError:
        inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
        c = b - inv_phi * (b - a)
        d = a + inv_phi * (b - a)
        fc, fd = func(c), func(d)
        for _ in range(maxiter):
            if abs(b - a) <= xtol:
                break
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - inv_phi * (b - a)
                fc = func(c)
            else:
                a, c, fc = c, d, fd
                d = a + inv_phi * (b - a)
                fd = func(d)
        return (a + b) / 2.0

    result = minimize_scalar(
        func,
        bounds=(a, b),
        method="bounded",
        options={"xatol": xtol, "maxiter": maxiter},
    )
    return float(result.x)


@dataclass
class ObstacleDistanceResult:
    """Minimum distance between a polyline obstacle and its span.

    Attributes:
        obstacle_name: Name of the obstacle.
        span_index: Index of the span the obstacle belongs to.
        distance_result: `DistanceResult` at the closest obstacle location.
            `distance_result.point_base` is the closest point on the obstacle.
        segment_index: Index of the obstacle segment holding the minimum
            (segment ``i`` joins points ``i`` and ``i + 1``). Always ``0``
            for single-point obstacles.
        segment_parameter: Position along the segment, in ``[0, 1]``
            (``0`` at point ``segment_index``, ``1`` at point
            ``segment_index + 1``).
    """

    obstacle_name: str
    span_index: int
    distance_result: DistanceResult
    segment_index: int
    segment_parameter: float

    @property
    def distance_3d(self) -> float:
        """Minimum distance between the obstacle and the span."""
        return self.distance_result.distance_3d

    @property
    def closest_obstacle_point(self) -> np.ndarray:
        """Closest point on the obstacle (absolute section frame)."""
        return self.distance_result.point_base

    @property
    def closest_span_point(self) -> np.ndarray:
        """Closest point on the span (absolute section frame)."""
        return self.distance_result.point_target


def plane_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of `planes.plane_from_line` for several normals.

    Args:
        normals: Unit horizontal normals of shape (S, 3).

    Returns:
        A tuple `(u, v)` of plane basis vectors, both of shape (S, 3).
    """
    normals = np.atleast_2d(np.asarray(normals, dtype=float))
    n0, n1 = normals[:, 0], normals[:, 1]
    u = np.column_stack([-n1, n0, np.zeros_like(n0)])
    # guard for vertical normals (n0 and n1 close to zero)
    vertical = (np.abs(n0) < 1e-6) & (np.abs(n1) < 1e-6)
    u[vertical] = [1.0, 0.0, 0.0]
    # normalize u to unit length
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    # create v and normalize to unit length
    v = np.cross(normals, u)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return u, v


def batch_plane_distances(
    points: np.ndarray,
    span_indices: np.ndarray,
    curves: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Plane distance of N points, each against the curve of its own span.

    For point ``p`` of span ``s``, the plane is the one through ``p`` with
    normal ``normals[s]``. The curve crossing is found from the two curve
    points closest to the plane, then linearly interpolated.

    Args:
        points: Section-frame points of shape (N, 3), or (3,) for one point.
        span_indices: Span index of each point, shape (N,).
        curves: Section-frame curve points of every span, shape (S, M, 3).
        normals: Unit horizontal plane normals of every span, shape (S, 3).

    Returns:
        A tuple `(distances, targets)` with shapes (N,) and (N, 3).
        Entries without intersection are `np.inf` and `np.nan`.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    span_indices = np.atleast_1d(np.asarray(span_indices, dtype=int))
    curves = np.asarray(curves, dtype=float)
    normals = np.asarray(normals, dtype=float)
    rows = np.arange(points.shape[0])

    point_normals = normals[span_indices]
    # project points and curves onto the plane defined by the normals
    # reminder: smk, sm -> sm means dots along the normal (s,3) for each curve point (s,m,3)
    points_proj = np.einsum("nk,nk->n", points, point_normals)
    curves_proj = np.einsum("smk,sk->sm", curves, normals)[span_indices]
    signed = curves_proj - points_proj[:, np.newaxis]

    # find the two curve points closest to the plane for each point
    closest = np.argsort(np.abs(signed), axis=1)[:, :2]
    closest.sort(axis=1)
    i0, i1 = closest[:, 0], closest[:, 1]

    crossing = signed[rows, i0] * signed[rows, i1] <= 0
    start = curves[span_indices, i0]
    segment = curves[span_indices, i1] - start
    # project the segment onto the plane defined by the point normals
    denominator = np.einsum("nk,nk->n", segment, point_normals)
    # check denominator to avoid division by zero (crossing + segment not parallel to plane)
    valid = crossing & (np.abs(denominator) > 1e-10)
    safe_denominator = np.where(valid, denominator, 1.0)

    # direct solving for the intersection parameter t along the segment
    # c(t) = start + t * segment
    # intersection with the plane occurs when dot(c(t), point_normals) = points_proj
    # solve for t: dot(start + t * segment, point_normals) = points_proj
    t = (
        points_proj - np.einsum("nk,nk->n", start, point_normals)
    ) / safe_denominator
    valid &= (t >= 0) & (t <= 1)

    # now you have t, let's find targets = start + t * segment
    targets = start + t[:, np.newaxis] * segment
    # compute distances from points to their corresponding targets on the curve along the plane
    distances = np.linalg.norm(targets - points, axis=1)
    # clear distances for invalid intersections
    distances = np.where(valid, distances, np.inf)
    targets = np.where(valid[:, np.newaxis], targets, np.nan)
    return distances, targets


class DistanceEngine:
    """DistanceEngine distance computation between a point and a curve in 3D space.

    It uses a defined plane for the distance calculation. The plane is defined by a span frame, which is determined by two points (start and end of the span). The engine allows to add curves and span frames, and then compute the distance from a given point to the curve along the plane defined by the span frame. The result is returned as a DistanceResult object containing the distance information and projection details.

    Examples:
        >>> de = DistanceEngine()
        >>> de.add_curves(curve_points)
        >>> de.add_span_frame(span_start, span_end)
        >>> distance_result = de.plane_distance(obstacle_point)
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

    # TODO: in the long run, "span" should not be an option: always in absolute coordinates
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

    def plane_distances(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized plane distance for N points given in the section frame.

        Requires `add_span_frame` and `add_curves` to have been called. Unlike
        `plane_distance`, this method does not modify the engine state.

        Args:
            points: A numpy array of shape (N, 3), or (3,) for a single point.

        Returns:
            A tuple `(distances_3d, targets)` with shapes (N,) and (N, 3).
            Entries without intersection are `np.inf` and `np.nan`.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return batch_plane_distances(
            points,
            span_indices=np.zeros(points.shape[0], dtype=int),
            curves=np.asarray(self.curve_points, dtype=float)[np.newaxis],
            normals=self.line_direction_normalized[np.newaxis],
        )
