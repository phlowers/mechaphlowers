# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np

from mechaphlowers.core.geometry.planes import (
    change_local_frame,
    intersection_curve_plane,
    plane_basis,
    plane_from_line,
)
from mechaphlowers.core.geometry.points import rotate_vector
from mechaphlowers.core.numeric.methods import golden_section_search
from mechaphlowers.entities.errors import NoIntersectionPlaneWarning

logger = logging.getLogger(__name__)


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


def build_distance_results(
    points: np.ndarray,
    span_indices: np.ndarray,
    targets: np.ndarray,
    distances: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> list[DistanceResult]:
    """Build one `DistanceResult` per row from batched plane distances.

    Args:
        points: Section-frame points, shape (N, 3).
        span_indices: Span index of each point, shape (N,).
        targets: Intersection points on the span, shape (N, 3).
        distances: 3D distances, shape (N,).
        u: Plane u basis of every span, shape (S, 3).
        v: Plane v basis of every span, shape (S, 3).
    """
    # increase u v size from S,3 to N,3
    u_rows = u[span_indices]
    v_rows = v[span_indices]

    diff = targets - points
    # project the vector onto u,v
    proj_u = np.einsum("nk,nk->n", diff, u_rows)
    proj_v = np.einsum("nk,nk->n", diff, v_rows)
    return [
        DistanceResult(
            point_base=points[i].copy(),
            point_target=targets[i].copy(),
            u_plane=u_rows[i].copy(),
            v_plane=v_rows[i].copy(),
            distance_3d=float(distances[i]),
            distance_projection_u=float(proj_u[i]),
            distance_projection_v=float(proj_v[i]),
        )
        for i in range(len(points))
    ]


def _validate_sampling(step: float, xtol: float) -> None:
    if step <= 0:
        raise ValueError("step must be strictly positive")
    if xtol <= 0:
        raise ValueError("xtol must be strictly positive")


class _SearchInterval(NamedTuple):
    obstacle_id: int
    segment_id: int
    lower: float
    upper: float


class _BestSamples(NamedTuple):
    seg: np.ndarray
    t: np.ndarray
    distance: np.ndarray

    def points(self, seg_start: np.ndarray, seg_vec: np.ndarray) -> np.ndarray:
        """Return the best point of each obstacle, shape ``(N, 3)``."""
        return seg_start[self.seg] + self.t[:, np.newaxis] * seg_vec[self.seg]


class _RefinedCandidates(NamedTuple):
    obstacle_id: np.ndarray
    segment_id: np.ndarray
    t: np.ndarray
    distance: np.ndarray


class _SegmentSamples(NamedTuple):
    seg_start: np.ndarray
    seg_vec: np.ndarray
    seg_loc: np.ndarray
    seg_sub: np.ndarray
    seg_span: np.ndarray
    seg_len: np.ndarray
    sample_seg: np.ndarray
    sample_t: np.ndarray
    sample_obs: np.ndarray
    spans: list[int]


def _build_segments_samples(
    coords_dict: dict[str, np.ndarray],
    spans_dict: dict[str, int],
    step: float,
) -> _SegmentSamples:
    """Split the obstacles into segments and sample them.

    Each obstacle polyline with `V_i` vertices yields `V_i - 1` segments
    (a single-vertex obstacle yields one zero-length segment). Each segment
    is subdivided into `seg_sub` pieces of at most `step` meters, and
    samples are placed at local parameters `t` in `[0, 1]`, excluding the
    end vertex of every segment except the last one of its obstacle.

    Args:
        coords_dict: Obstacle name to vertices, each of shape `(V_i, 3)`.
            Its order defines the obstacle ids.
        spans_dict: Obstacle name to its span index.
        step: Maximum sampling step along the obstacle segments, in meters.

    Returns:
        A `_SegmentSamples` with, for `K = sum(V_i - 1)` segments:
            - `seg_start`, `seg_vec`: segment start points and vectors,
              shape `(K, 3)`.
            - `seg_loc`: index of the segment within its obstacle, `(K,)`.
            - `seg_sub`: number of subdivisions of each segment, `(K,)`.
            - `seg_span`: span index of each segment, `(K,)`.
            - `seg_len`: length of each segment in meters, `(K,)`.
        and, for `P` samples in total:
            - `sample_seg`: segment index of each sample, `(P,)`.
            - `sample_t`: local segment parameter of each sample, `(P,)`.
            - `sample_obs`: obstacle index of each sample, `(P,)`.
        and `spans`: span index of each of the `N` obstacles, as a list.
    """
    seg_starts: list[np.ndarray] = []
    seg_vectors: list[np.ndarray] = []
    seg_obstacle: list[np.ndarray] = []
    seg_local: list[np.ndarray] = []
    seg_nb_sub: list[np.ndarray] = []
    sample_segments: list[np.ndarray] = []
    sample_ts: list[np.ndarray] = []
    spans: list[int] = []

    n_segments = 0
    for obstacle_id, name in enumerate(coords_dict):
        vertices = np.asarray(coords_dict[name], dtype=float)
        spans.append(int(spans_dict[name]))
        if len(vertices) == 1:
            starts, vectors = vertices, np.zeros_like(vertices)
        else:
            starts, vectors = vertices[:-1], np.diff(vertices, axis=0)
        nb_segments = len(starts)
        nb_subdivisions = np.maximum(
            np.ceil(np.linalg.norm(vectors, axis=1) / step), 1
        ).astype(int)
        for local in range(nb_segments):
            ts = np.linspace(0.0, 1.0, nb_subdivisions[local] + 1)
            if len(vertices) == 1:
                ts = ts[:1]
            elif local < nb_segments - 1:
                ts = ts[:-1]
            sample_segments.append(np.full(len(ts), n_segments + local))
            sample_ts.append(ts)
        seg_starts.append(starts)
        seg_vectors.append(vectors)
        seg_obstacle.append(np.full(nb_segments, obstacle_id))
        seg_local.append(np.arange(nb_segments))
        seg_nb_sub.append(nb_subdivisions)
        n_segments += nb_segments

    seg_start = np.concatenate(seg_starts)
    seg_vec = np.concatenate(seg_vectors)
    seg_obs = np.concatenate(seg_obstacle)
    seg_loc = np.concatenate(seg_local)
    seg_sub = np.concatenate(seg_nb_sub)
    seg_span = np.asarray(spans)[seg_obs]
    seg_len = np.linalg.norm(seg_vec, axis=1)
    sample_seg = np.concatenate(sample_segments)
    sample_t = np.concatenate(sample_ts)
    sample_obs = seg_obs[sample_seg]
    return _SegmentSamples(
        seg_start,
        seg_vec,
        seg_loc,
        seg_sub,
        seg_span,
        seg_len,
        sample_seg,
        sample_t,
        sample_obs,
        spans,
    )


def _warning_no_intersection(
    names: list[str],
    spans: list[int],
    n_samples: np.ndarray,
    n_skipped: np.ndarray,
) -> None:
    """Log and warn for each obstacle having samples without intersection.

    Args:
        names: Obstacle names, indexed by obstacle id.
        spans: Span index of each obstacle, indexed by obstacle id.
        n_samples: Number of sampled points per obstacle.
        n_skipped: Number of sampled points without intersection per obstacle.
    """
    for obstacle_id in map(int, np.flatnonzero(n_skipped)):
        message = (
            f"No intersection found between span {spans[obstacle_id]} and "
            f"{n_skipped[obstacle_id]}/{n_samples[obstacle_id]} sampled points of obstacle "
            f"'{names[obstacle_id]}' - skipping them."
        )
        logger.warning(message)
        # points to the caller of PositionEngine's public method
        warnings.warn(message, NoIntersectionPlaneWarning, stacklevel=6)


def _segment_plane_distances(
    segments: _SegmentSamples,
    seg_ids: np.ndarray,
    t: np.ndarray,
    curves: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """Plane distances of points lying on segments.

    Args:
        segments: Segment and sample description of the obstacles.
        seg_ids: Segment index of each point, shape ``(N,)``.
        t: Local segment parameter of each point, shape ``(N,)``.
        curves: Section-frame curve points of every span, shape (S, M, 3).
        normals: Unit horizontal plane normals of every span, shape (S, 3).

    Returns:
        Distances of the points, shape ``(N,)``.
    """
    points = (
        segments.seg_start[seg_ids]
        + t[:, np.newaxis] * segments.seg_vec[seg_ids]
    )
    distances, _ = batch_plane_distances(
        points, segments.seg_span[seg_ids], curves, normals
    )
    return distances


def _coarse_sampling(
    names: list[str],
    segments: _SegmentSamples,
    curves: np.ndarray,
    normals: np.ndarray,
) -> _BestSamples:
    """Evaluate every sample and keep the closest one of each obstacle.

    Obstacles having samples without intersection are reported with a warning.
    On ties, the first sample is kept.
    """
    sample_obs = segments.sample_obs
    sample_distances = _segment_plane_distances(
        segments, segments.sample_seg, segments.sample_t, curves, normals
    )

    n_obstacles = len(names)
    n_samples = np.bincount(sample_obs, minlength=n_obstacles)
    n_skipped = np.bincount(
        sample_obs[~np.isfinite(sample_distances)], minlength=n_obstacles
    )
    _warning_no_intersection(names, segments.spans, n_samples, n_skipped)

    order = np.lexsort((sample_distances, sample_obs))
    group_start = np.searchsorted(sample_obs[order], np.arange(n_obstacles))
    best = order[group_start]
    return _BestSamples(
        seg=segments.sample_seg[best],
        t=segments.sample_t[best],
        distance=sample_distances[best],
    )


def _keep_better_candidates(
    best_samples: _BestSamples,
    candidates: _RefinedCandidates,
) -> None:
    """Update each obstacle's best sample when a refined candidate is better.

    Updates the arrays of `best_samples` in place. A candidate
    replaces the current best only if its distance is strictly smaller, so
    NaN candidates and ties are ignored. Candidates are processed in order,
    so several intervals of the same obstacle compete with each other.
    """
    for obs_id, segment_id, t_refined, distance in zip(
        candidates.obstacle_id,
        candidates.segment_id,
        candidates.t,
        candidates.distance,
    ):
        if distance < best_samples.distance[obs_id]:
            best_samples.seg[obs_id] = segment_id
            best_samples.t[obs_id] = t_refined
            best_samples.distance[obs_id] = distance


def _assemble_obstacle_results(
    names: list[str],
    spans: list[int],
    seg_loc: np.ndarray,
    best_samples: _BestSamples,
    found: np.ndarray,
    distance_results: list[DistanceResult],
) -> dict[str, ObstacleDistanceResult | None]:
    """Map the found distance results back to their obstacle names.

    `found` holds the obstacle indices of `distance_results`, in the same
    order. Obstacles absent from `found` keep `None`.
    """
    results: dict[str, ObstacleDistanceResult | None] = {
        name: None for name in names
    }
    for obstacle_id, distance_result in zip(found.tolist(), distance_results):
        results[names[obstacle_id]] = ObstacleDistanceResult(
            obstacle_name=names[obstacle_id],
            span_index=spans[obstacle_id],
            distance_result=distance_result,
            segment_index=int(seg_loc[best_samples.seg[obstacle_id]]),
            segment_parameter=float(best_samples.t[obstacle_id]),
        )
    return results


def search_interval_refinement(
    best_samples: _BestSamples,
    seg_nb_vertices: np.ndarray,
    seg_sub: np.ndarray,
    seg_loc: np.ndarray,
    seg_len: np.ndarray,
) -> list[_SearchInterval]:
    """Build the golden-section search intervals around each coarse minimum.

    Returns:
        One interval per refined obstacle, plus an extra interval on the
        previous segment when the minimum sits on a shared vertex. Intervals
        on zero-length segments are dropped.
    """
    intervals: list[_SearchInterval] = []
    obstacle_id_to_refine = np.flatnonzero(
        np.isfinite(best_samples.distance) & (seg_nb_vertices > 1)
    )  # refine only obstacles with finite distance and more than one segment vertex
    for obstacle_id in map(int, obstacle_id_to_refine):
        segment_id = int(best_samples.seg[obstacle_id])
        t0 = float(best_samples.t[obstacle_id])

        # add/subtract dt to create the search interval around t0
        dt = 1.0 / seg_sub[segment_id]  # warning dt is in parameter unit

        # Search both sides of a shared vertex.
        intervals.append(
            _SearchInterval(
                obstacle_id=obstacle_id,
                segment_id=segment_id,
                lower=max(t0 - dt, 0.0),
                upper=min(t0 + dt, 1.0),
            )
        )
        # If t0 == 0 and this isn't the obstacle's first segment, the minimum may sit on the shared vertex with the previous segment. So an extra interval is added on segment_id - 1, covering [1 - prev_dt, 1]
        if t0 == 0.0 and seg_loc[segment_id] > 0:
            prev_dt = 1.0 / seg_sub[segment_id - 1]
            intervals.append(
                _SearchInterval(
                    obstacle_id=obstacle_id,
                    segment_id=segment_id - 1,
                    lower=1.0 - prev_dt,
                    upper=1.0,
                )
            )

    # compute valid intervals with positive segment length where search is meaningful
    return [iv for iv in intervals if seg_len[iv.segment_id] > 0.0]


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

    def min_distances(
        self,
        coords_dict: dict[str, np.ndarray],
        spans_dict: dict[str, int],
        curves: np.ndarray,
        normals: np.ndarray,
        step: float,
        refine: bool,
        xtol: float,
    ) -> dict[str, ObstacleDistanceResult | None]:
        """Minimum distance of several polyline obstacles, computed together.

        All samples of all obstacles are evaluated in one batch. When `refine`
        is `True`, the minimum around each obstacle best sample is refined by
        a golden-section search, run simultaneously for all obstacles. This
        method does not modify the engine state.

        Shapes used internally, with `N` obstacles, `S` spans,
        `V_i` vertices of obstacle `i`, `K = sum(V_i - 1)` segments (a
        single-vertex obstacle counts as one zero-length segment) and
        `P` samples in total:
            - obstacle polyline `vertices`: (V_i, 3)
            - segment starts and vectors: (K, 3)
            - segment subdivisions: (K,)
            - sample segment indices and parameters `t`: (P,)
            - span curves: (S, M, 3); span normals, `u` and `v` bases: (S, 3)
            - best sample per obstacle, before and after refinement: (N,)

        Args:
            coords_dict: Obstacle name to vertices in the section frame, each
                of shape (V_i, 3).
            spans_dict: Obstacle name to the index of its span.
            curves: Section-frame curve points of every span, shape (S, M, 3).
            normals: Unit horizontal plane normals of every span, shape (S, 3).
            step: Sampling step along the obstacle segments, in meters.
            refine: Refine the minimum between samples.
            xtol: Absolute tolerance of the refinement, in meters.

        Returns:
            Dict mapping each of the `N` obstacle names to its
            `ObstacleDistanceResult`, or `None` if no sample of the obstacle
            faces the span.

        Raises:
            ValueError: If `step` or `xtol` is not strictly positive.
        """
        _validate_sampling(step, xtol)
        if not coords_dict:
            return {}

        names = list(coords_dict)
        u, v = plane_basis(normals)

        segments_samples = _build_segments_samples(
            coords_dict, spans_dict, step
        )
        seg_start = segments_samples.seg_start
        seg_vec = segments_samples.seg_vec
        seg_loc = segments_samples.seg_loc
        seg_sub = segments_samples.seg_sub
        seg_span = segments_samples.seg_span
        seg_len = segments_samples.seg_len
        spans = segments_samples.spans

        best_samples = _coarse_sampling(
            names, segments_samples, curves, normals
        )

        if refine:
            # 2 steps refinement:
            # - search interval to find the best little segment and avoid multiple local minima
            #   it is cheap because vectorized -> golden search can quickly converge after
            # - golden search (need several iterations to converge)

            seg_nb_vertices = np.array([len(coords_dict[n]) for n in names])

            intervals = search_interval_refinement(
                best_samples, seg_nb_vertices, seg_sub, seg_loc, seg_len
            )

            if intervals:
                # convert into numpy arrays for vectorized processing
                obs_ids = np.array(
                    [iv.obstacle_id for iv in intervals], dtype=int
                )
                interval_segs = np.array(
                    [iv.segment_id for iv in intervals], dtype=int
                )
                lower = np.array([iv.lower for iv in intervals])
                upper = np.array([iv.upper for iv in intervals])

                refined_t = golden_section_search(
                    lambda t: _segment_plane_distances(
                        segments_samples, interval_segs, t, curves, normals
                    ),
                    lower,
                    upper,
                    tol=xtol
                    / seg_len[
                        interval_segs
                    ],  # unit: relative to segment length
                )

                refined_distance = _segment_plane_distances(
                    segments_samples, interval_segs, refined_t, curves, normals
                )

                _keep_better_candidates(
                    best_samples,
                    _RefinedCandidates(
                        obstacle_id=obs_ids,
                        segment_id=interval_segs,
                        t=refined_t,
                        distance=refined_distance,
                    ),
                )

        best_points = best_samples.points(seg_start, seg_vec)
        best_spans = seg_span[best_samples.seg]
        final_distances, final_targets = batch_plane_distances(
            best_points, best_spans, curves, normals
        )

        # Filter out invalid distances and build the final distance results for valid samples
        found = np.flatnonzero(np.isfinite(final_distances))
        distance_results = build_distance_results(
            best_points[found],
            best_spans[found],
            final_targets[found],
            final_distances[found],
            u,
            v,
        )

        return _assemble_obstacle_results(
            names, spans, seg_loc, best_samples, found, distance_results
        )
