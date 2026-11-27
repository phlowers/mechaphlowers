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
    plane_basis,
    plane_from_line,
    scalar_intersection_curve_plane,
)
from mechaphlowers.core.geometry.points import rotate_vector
from mechaphlowers.core.numeric.methods import golden_section_search
from mechaphlowers.entities.errors import NoIntersectionPlaneWarning

logger = logging.getLogger(__name__)


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

    def scalar_define_distance_plane(
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
    def scalar_plane_distance(
        self,
        point_base: np.ndarray,
        frame: Literal["span", "section"] = "span",
    ) -> PointDistanceResult:
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
        self.scalar_define_distance_plane(point=self.point_base)

        intersection_point = scalar_intersection_curve_plane(
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
            scalar_points_distance_inside_plane(
                point_base=self.point_base,
                point_target=intersection_point[0],
                u_plane=self.u_plane,
                v_plane=self.v_plane,
            )
        )

        return PointDistanceResult(
            point_base=self.point_base,
            point_target=intersection_point[0],
            u_plane=self.u_plane,
            v_plane=self.v_plane,
            distance_3d=distance_3d,
            distance_projection_u=distance_projection_u,
            distance_projection_v=distance_projection_v,
        )


def _span_normals(ground_supports: np.ndarray) -> np.ndarray:
    """Unit horizontal normals of the spans between consecutive ground supports.

    Args:
        ground_supports: Ground coordinates of the supports, shape (S+1, 3).

    Returns:
        Unit normals of shape (S, 3), with zero vertical component.
    """
    ground = np.array(ground_supports, dtype=float)
    # project ground points onto the XY plane
    ground[:, 2] = 0.0
    # get the direction vectors of the spans in the XY plane
    axis = np.diff(ground, axis=0)
    # normalize the direction vectors to get horizontal normals
    return axis / np.linalg.norm(axis, axis=1, keepdims=True)


@dataclass(frozen=True)
class SamplingOptions:
    """Default sampling options of `VectorizedDistanceEngine`.

    Attributes:
        step: Sampling step along the obstacle segments, in meters.
        refine: Refine the minimum between samples.
        xtol: Absolute tolerance of the refinement, in meters.

    Raises:
        ValueError: If `step` or `xtol` is not strictly positive.
    """

    step: float = 1.0
    refine: bool = True
    xtol: float = 1e-3

    def __post_init__(self) -> None:
        _validate_sampling(self.step, self.xtol)


class VectorizedDistanceEngine:
    """Plane distances of obstacles to several spans, computed in batch.

    The engine holds a mutable store: the unit normals and `u`, `v` plane
    bases of every span (see `set_frames`), and the section-frame curve
    points of every span (see `set_curves` and `update_spans`). Distance
    methods read this store as it is; they never refresh it.

    Args:
        ground_supports: Ground coordinates of the supports, shape (S+1, 3).
            Optional, see `set_frames`.
        curves: Section-frame curve points of every span, shape (S, M, 3).
            Optional, see `set_curves`.
        options: Default sampling options of the distance methods.
            Defaults to `SamplingOptions()`.

    Examples:
        >>> engine = VectorizedDistanceEngine(ground_supports, curves)
        >>> engine.update_spans([2], new_curves_of_span_2)
        >>> engine.min_distances(coords_dict, spans_dict)
    """

    def __init__(
        self,
        ground_supports: np.ndarray | None = None,
        curves: np.ndarray | None = None,
        options: SamplingOptions | None = None,
    ) -> None:
        self.options = options if options is not None else SamplingOptions()
        self.normals: np.ndarray | None = None
        self.u: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.curves: np.ndarray | None = None
        self.curves_proj: np.ndarray | None = None
        if ground_supports is not None:
            self.set_frames(ground_supports)
        if curves is not None:
            self.set_curves(curves)

    def set_frames(self, ground_supports: np.ndarray) -> None:
        """Set the frame of every span from the ground supports.

        Recomputes the normals and plane bases, and the projected curves if
        curves are already set. Changing the frames is rare: prefer
        `update_spans` for changes of the curves only.

        Args:
            ground_supports: Ground coordinates of the supports, shape (S+1, 3).

        Raises:
            ValueError: If the shape is not (S+1, 3) with S >= 1, or if the
                curves already set do not have S spans.
        """
        ground_supports = np.asarray(ground_supports, dtype=float)
        if ground_supports.ndim != 2 or ground_supports.shape[1] != 3:
            raise ValueError("ground_supports must have shape (S+1, 3)")
        if ground_supports.shape[0] < 2:
            raise ValueError("ground_supports must contain at least 2 points")
        normals = _span_normals(ground_supports)
        if self.curves is not None and self.curves.shape[0] != len(normals):
            raise ValueError(
                f"ground_supports define {len(normals)} spans but curves "
                f"define {self.curves.shape[0]}"
            )
        self.normals = normals
        self.u, self.v = plane_basis(normals)
        self._refresh_curves_proj()

    def set_curves(self, curves: np.ndarray) -> None:
        """Set the section-frame curve points of all spans.

        The array is copied, so that later changes of the caller's array do
        not affect the engine. Use `update_spans` to change some spans.

        Args:
            curves: Section-frame curve points of every span, shape (S, M, 3).

        Raises:
            ValueError: If the shape is not (S, M, 3), or if the frames
                already set do not have S spans.
        """
        curves = np.array(curves, dtype=float)
        if curves.ndim != 3 or curves.shape[2] != 3:
            raise ValueError("curves must have shape (S, M, 3)")
        if self.normals is not None and curves.shape[0] != len(self.normals):
            raise ValueError(
                f"curves have {curves.shape[0]} spans but frames define "
                f"{len(self.normals)}"
            )
        self.curves = curves
        self._refresh_curves_proj()

    def set_spans(
        self, ground_supports: np.ndarray, curves: np.ndarray
    ) -> None:
        """Set the frames and the curves of all spans together.

        Unlike calling `set_frames` then `set_curves`, the number of spans
        may change (e.g. after a section change).

        Args:
            ground_supports: Ground coordinates of the supports, shape (S+1, 3).
            curves: Section-frame curve points of every span, shape (S, M, 3).

        Raises:
            ValueError: If the shapes are invalid or the span counts differ.
        """
        self.normals = self.u = self.v = None
        self.curves = self.curves_proj = None
        self.set_frames(ground_supports)
        self.set_curves(curves)

    def update_spans(self, span_ids: np.ndarray, curves: np.ndarray) -> None:
        """Replace the curve points of some spans, in place.

        Only the given rows of the curves and of their projections are
        updated: the cost is proportional to the number of updated spans.

        Args:
            span_ids: Indices of the spans to update, shape (k,).
            curves: New section-frame curve points of these spans, shape
                (k, M, 3), with the same M as the curves already set.

        Raises:
            RuntimeError: If `set_frames` and `set_curves` were not both called.
            IndexError: If a span index is out of range.
            ValueError: If the shape of `curves` does not match `span_ids`.
        """
        self._require_ready()
        span_ids = np.atleast_1d(np.asarray(span_ids, dtype=int))
        curves = np.asarray(curves, dtype=float)
        n_spans, n_points, _ = self.curves.shape
        if span_ids.ndim != 1:
            raise ValueError("span_ids must be a 1D sequence of indices")
        if span_ids.size and (span_ids.min() < 0 or span_ids.max() >= n_spans):
            raise IndexError(
                f"span_ids must be in [0, {n_spans - 1}], got {span_ids}"
            )
        if curves.shape != (span_ids.size, n_points, 3):
            raise ValueError(
                f"curves must have shape ({span_ids.size}, {n_points}, 3), "
                f"got {curves.shape}"
            )
        self.curves[span_ids] = curves
        self.curves_proj[span_ids] = np.einsum(
            "smk,sk->sm", curves, self.normals[span_ids]
        )

    def _refresh_curves_proj(self) -> None:
        """Recompute the projections of all curves, when both are set."""
        if self.curves is None or self.normals is None:
            self.curves_proj = None
        else:
            self.curves_proj = np.einsum(
                "smk,sk->sm", self.curves, self.normals
            )

    def _require_ready(self) -> None:
        if self.normals is None or self.curves is None:
            raise RuntimeError(
                "Distances need the span frames and curves: call set_frames "
                "and set_curves (or pass ground_supports and curves) first."
            )

    def _resolve_sampling(
        self,
        step: float | None,
        refine: bool | None,
        xtol: float | None,
    ) -> tuple[float, bool, float]:
        """Replace the `None` arguments by the default options, then validate."""
        step = self.options.step if step is None else step
        refine = self.options.refine if refine is None else refine
        xtol = self.options.xtol if xtol is None else xtol
        _validate_sampling(step, xtol)
        return step, refine, xtol

    def plane_distances(
        self, points: np.ndarray, span_indices: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized plane distance of N points, each to its own span.

        Args:
            points: A numpy array of shape (N, 3), or (3,) for a single point,
                in the section frame.
            span_indices: Span index of each point, shape (N,). Defaults to
                span 0 for all points.

        Returns:
            A tuple `(distances_3d, targets)` with shapes (N,) and (N, 3).
            Entries without intersection are `np.inf` and `np.nan`.

        Raises:
            RuntimeError: If `set_frames` and `set_curves` were not both called.
        """
        self._require_ready()
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if span_indices is None:
            span_indices = np.zeros(points.shape[0], dtype=int)
        return batch_plane_distances(
            points,
            span_indices=span_indices,
            curves=self.curves,
            normals=self.normals,
            curves_proj=self.curves_proj,
        )

    def min_distances(
        self,
        coords_dict: dict[str, np.ndarray],
        spans_dict: dict[str, int],
        step: float | None = None,
        refine: bool | None = None,
        xtol: float | None = None,
    ) -> dict[str, ClosestDistanceResult | None]:
        """Minimum distance of several polyline obstacles, computed together.

        Shortcut for the `minimum` of
        [`distances`][mechaphlowers.core.geometry.distances.VectorizedDistanceEngine.distances]
        with polyline obstacles.

        Args:
            coords_dict: Obstacle name to vertices in the section frame.
            spans_dict: Obstacle name to the index of its span.
            step: Sampling step, in meters. `None` uses `options.step`.
            refine: Refine the minimum between samples. `None` uses
                `options.refine`.
            xtol: Absolute tolerance of the refinement, in meters. `None`
                uses `options.xtol`.

        Returns:
            Dict mapping each obstacle name to its `ClosestDistanceResult`,
            or `None` if no sample of the obstacle faces the span.

        Raises:
            RuntimeError: If `set_frames` and `set_curves` were not both called.
            ValueError: If `step` or `xtol` is not strictly positive.
        """
        return {
            name: obstacle_distances.minimum
            for name, obstacle_distances in self.distances(
                coords_dict,
                spans_dict,
                step=step,
                refine=refine,
                xtol=xtol,
            ).items()
        }

    def distances(
        self,
        coords_dict: dict[str, np.ndarray],
        spans_dict: dict[str, int],
        geometry_types: dict[str, Literal["points", "polyline"]] | None = None,
        step: float | None = None,
        refine: bool | None = None,
        xtol: float | None = None,
    ) -> dict[str, ObstacleDistances]:
        """Distances of several obstacles, computed together.

        Every obstacle vertex gets its plane distance to its span. The
        minimum of each obstacle is searched among its samples: every vertex,
        plus, for a polyline, samples every `step` meters along its segments.
        All samples of all obstacles are evaluated in one batch. When `refine`
        is `True`, the minimum around each polyline best sample is refined by
        a golden-section search, run simultaneously for all obstacles. This
        method does not modify the engine state.

        Shapes used internally, with `N` obstacles, `S` spans,
        `V_i` vertices of obstacle `i`, `K` segments (`V_i - 1` for a
        polyline, `V_i` zero-length segments for discrete points or a
        single-vertex obstacle) and `P` samples in total:
            - obstacle `vertices`: (V_i, 3)
            - segment starts and vectors: (K, 3)
            - segment subdivisions: (K,)
            - sample segment indices and parameters `t`: (P,)
            - span curves: (S, M, 3); span normals, `u` and `v` bases: (S, 3)
            - best sample per obstacle, before and after refinement: (N,)

        Args:
            coords_dict: Obstacle name to vertices in the section frame, each
                of shape (V_i, 3), in `point_index` order.
            spans_dict: Obstacle name to the index of its span.
            geometry_types: Obstacle name to ``"points"`` or ``"polyline"``.
                Missing obstacles are polylines. Defaults to `None`.
            step: Sampling step along the obstacle segments, in meters.
                `None` uses `options.step`.
            refine: Refine the minimum between samples. `None` uses
                `options.refine`.
            xtol: Absolute tolerance of the refinement, in meters. `None`
                uses `options.xtol`.

        Returns:
            Dict mapping each of the `N` obstacle names to its
            `ObstacleDistances`. The `points` of each obstacle are keyed by
            vertex position within its own array of `coords_dict`.

        Raises:
            RuntimeError: If `set_frames` and `set_curves` were not both called.
            ValueError: If `step` or `xtol` is not strictly positive.
        """
        self._require_ready()
        step, refine, xtol = self._resolve_sampling(step, refine, xtol)
        if not coords_dict:
            return {}

        names = list(coords_dict)
        geometry_types = {
            name: (geometry_types or {}).get(name, "polyline")
            for name in names
        }

        segments_samples = _build_segments_samples(
            coords_dict, spans_dict, step, geometry_types
        )
        seg_loc = segments_samples.seg_loc
        seg_sub = segments_samples.seg_sub
        seg_span = segments_samples.seg_span
        seg_len = segments_samples.seg_len
        spans = segments_samples.spans

        coarse_samples = _coarse_sampling(
            names,
            segments_samples,
            self.curves,
            self.normals,
            self.curves_proj,
        )
        best_samples = coarse_samples.best

        if refine:
            # 2 steps refinement:
            # - search interval to find the best little segment and avoid multiple local minima
            #   it is cheap because vectorized -> golden search can quickly converge after
            # - golden search (need several iterations to converge)

            # discrete points have no segment to refine
            seg_nb_vertices = np.array(
                [
                    len(coords_dict[n])
                    if geometry_types[n] == "polyline"
                    else 1
                    for n in names
                ]
            )

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
                        segments_samples,
                        interval_segs,
                        t,
                        self.curves,
                        self.normals,
                        self.curves_proj,
                    ),
                    lower,
                    upper,
                    tol=xtol
                    / seg_len[
                        interval_segs
                    ],  # unit: relative to segment length
                )

                refined_distance = _segment_plane_distances(
                    segments_samples,
                    interval_segs,
                    refined_t,
                    self.curves,
                    self.normals,
                    self.curves_proj,
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

        best_points = segments_samples.points(best_samples.seg, best_samples.t)
        best_spans = seg_span[best_samples.seg]
        final_distances, final_targets = batch_plane_distances(
            best_points,
            best_spans,
            self.curves,
            self.normals,
            curves_proj=self.curves_proj,
        )

        # Filter out invalid distances and build the final distance results for valid samples
        found = np.flatnonzero(np.isfinite(final_distances))
        distance_results = build_distance_results(
            best_points[found],
            best_spans[found],
            final_targets[found],
            final_distances[found],
            self.u,
            self.v,
        )

        minimums = _assemble_obstacle_results(
            names, spans, seg_loc, best_samples, found, distance_results
        )
        vertices = _vertex_results(
            segments_samples, coarse_samples, len(names), self.u, self.v
        )
        return {
            name: ObstacleDistances(
                obstacle_name=name,
                span_index=spans[obstacle_id],
                geometry_type=geometry_types[name],
                points=vertices[obstacle_id],
                minimum=minimums[name],
            )
            for obstacle_id, name in enumerate(names)
        }


def scalar_points_distance_inside_plane(
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


class PointDistanceResult:
    """Store a 3D point-to-point distance and its plane projections.

    Attributes:
        point_base: Base point as a 3D NumPy array.
        point_target: Target point as a 3D NumPy array.
        distance_3d: Distance between the two points in 3D.
        signed_distance_projection_u: Signed component along ``u_plane``.
        signed_distance_projection_v: Signed component along ``v_plane``.
        u_plane: First basis vector of the plane.
        v_plane: Second basis vector of the plane.
        distance_projection_u: Absolute value of
            ``signed_distance_projection_u``.
        distance_projection_v: Absolute value of
            ``signed_distance_projection_v``.

    Methods:
        __str__(): Format the points, distances, and plane vectors as text.
        __repr__(): Return a representation containing the formatted result.
        projection_points(origin_point): Return the points reached from
            ``origin_point`` by applying the signed ``u`` and ``v`` components.
        compute_new_frame(translation_vector, angle_to_project): Translate and
            rotate this result in place, then return it.
        generate_with_reversed_y_axis(): Return a deep copy with the y
            components of its points and plane vectors negated.

    The seven constructor arguments are the base and target points, plane basis
    vectors, 3D distance, and signed projection components.
    """

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
            f"PointDistanceResult:\n"
            f"  Point Base: ({self.point_base[0]:.4f}, {self.point_base[1]:.4f}, {self.point_base[2]:.4f})\n"
            f"  Point Target: ({self.point_target[0]:.4f}, {self.point_target[1]:.4f}, {self.point_target[2]:.4f})\n"
            f"  Distance 3D: {self.distance_3d:.6f} m\n"
            f"  Distance Projection U: {self.signed_distance_projection_u:.6f} m\n"
            f"  Distance Projection V: {self.signed_distance_projection_v:.6f} m\n"
            f"  U Plane Vector: ({self.u_plane[0]:.4f}, {self.u_plane[1]:.4f}, {self.u_plane[2]:.4f})\n"
            f"  V Plane Vector: ({self.v_plane[0]:.4f}, {self.v_plane[1]:.4f}, {self.v_plane[2]:.4f})\n"
        )

    def __repr__(self):
        return "<PointDistanceResult>\n" + self.__str__()

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
    ) -> PointDistanceResult:
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


# Backward-compatible alias of the former class name.
DistanceResult = PointDistanceResult


@dataclass
class ClosestDistanceResult:
    """Minimum distance between an obstacle and its span.

    Attributes:
        obstacle_name: Name of the obstacle.
        span_index: Index of the span the obstacle belongs to.
        distance_result: `DistanceResult` at the closest obstacle location.
            `distance_result.point_base` is the closest point on the obstacle.
        segment_index: For a polyline, index of the obstacle segment holding
            the minimum (segment ``i`` joins points ``i`` and ``i + 1``).
            For discrete points, `point_index` of the closest point.
            Always ``0`` for single-point obstacles.
        segment_parameter: Position along the segment, in ``[0, 1]``
            (``0`` at point ``segment_index``, ``1`` at point
            ``segment_index + 1``). Always ``0`` for discrete points.
    """

    obstacle_name: str
    span_index: int
    distance_result: PointDistanceResult
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

    def __str__(self):
        return (
            f"ClosestDistanceResult:\n"
            f"  Obstacle: {self.obstacle_name} (span {self.span_index})\n"
            f"  Segment: index {self.segment_index}, parameter {self.segment_parameter:.4f}\n"
            f"  Distance 3D: {self.distance_3d:.6f} m\n"
            f"  Closest obstacle point: ({self.closest_obstacle_point[0]:.4f}, {self.closest_obstacle_point[1]:.4f}, {self.closest_obstacle_point[2]:.4f})\n"
            f"  Closest span point: ({self.closest_span_point[0]:.4f}, {self.closest_span_point[1]:.4f}, {self.closest_span_point[2]:.4f})\n"
        )

    def __repr__(self):
        return "<ClosestDistanceResult>\n" + self.__str__()


@dataclass
class ObstacleDistances:
    """Distances between an obstacle and its span.

    Attributes:
        obstacle_name: Name of the obstacle.
        span_index: Index of the span the obstacle belongs to.
        geometry_type: ``"points"`` or ``"polyline"``.
        points: `point_index` order position of each obstacle vertex, to the
            `DistanceResult` of this point (`point_base` on the obstacle,
            `point_target` on the cable). Points whose distance plane does not
            intersect the span are absent.
        minimum: Closest obstacle location. For a polyline with at least two
            points, it is searched along the polyline and can lie between two
            points; otherwise it is the closest point. `None` if no point
            intersects the span.
    """

    obstacle_name: str
    span_index: int
    geometry_type: Literal["points", "polyline"]
    points: dict[int, PointDistanceResult]
    minimum: ClosestDistanceResult | None

    def compute_new_frame(
        self, translation_vector: np.ndarray, angle_to_project: np.float64
    ) -> ObstacleDistances:
        """Change the frame of every distance result, in place."""
        for distance_result in self.points.values():
            distance_result.compute_new_frame(
                translation_vector, angle_to_project
            )
        if self.minimum is not None:
            self.minimum.distance_result.compute_new_frame(
                translation_vector, angle_to_project
            )
        return self

    def get_table(self) -> np.ndarray:
        """Return a structured array with one row per obstacle point.

        Vector fields are split into scalar columns so the result converts
        directly with ``pandas.DataFrame(table)``.
        """
        table = []
        for point_index, distance_result in self.points.items():
            table.append(
                (
                    point_index,
                    distance_result.distance_3d,
                    *distance_result.point_base,
                    *distance_result.point_target,
                )
            )
        return np.array(
            table,
            dtype=[
                ("point_index", int),
                ("distance_3d", float),
                ("point_base_x", float),
                ("point_base_y", float),
                ("point_base_z", float),
                ("point_target_x", float),
                ("point_target_y", float),
                ("point_target_z", float),
            ],
        )

    def __str__(self):
        minimum_text = (
            "None"
            if self.minimum is None
            else f"{self.minimum.distance_3d:.6f} m (segment {self.minimum.segment_index}, parameter {self.minimum.segment_parameter:.4f})"
        )
        return (
            f"ObstacleDistances:\n"
            f"  Obstacle: {self.obstacle_name} (span {self.span_index})\n"
            f"  Geometry type: {self.geometry_type}\n"
            f"  Points with distance: {len(self.points)}\n"
            f"  Minimum: {minimum_text}\n"
        )

    def __repr__(self):
        return "<ObstacleDistances>\n" + self.__str__()

    def __len__(self):
        return len(self.points)


def batch_plane_distances(
    points: np.ndarray,
    span_indices: np.ndarray,
    curves: np.ndarray,
    normals: np.ndarray,
    curves_proj: np.ndarray | None = None,
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
        curves_proj: Projections of `curves` on `normals`, shape (S, M).
            Optional cache: computed from `curves` and `normals` when `None`.

    Returns:
        A tuple `(distances, targets)` with shapes (N,) and (N, 3).
        Entries without intersection are `np.inf` and `np.nan`.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    span_indices = np.atleast_1d(np.asarray(span_indices, dtype=int))
    curves = np.asarray(curves, dtype=float)
    normals = np.asarray(normals, dtype=float)
    rows = np.arange(points.shape[0])

    n_spans = len(normals)
    if span_indices.size and (
        span_indices.min() < 0 or span_indices.max() >= n_spans
    ):
        raise IndexError(
            f"span_indices must be in [0, {n_spans - 1}], got {span_indices}"
        )
    point_normals = normals[span_indices]
    # project points and curves onto the plane defined by the normals
    # reminder: smk, sm -> sm means dots along the normal (s,3) for each curve point (s,m,3)
    # TODO: draw
    points_proj = np.einsum("nk,nk->n", points, point_normals)
    if curves_proj is None:
        curves_proj = np.einsum("smk,sk->sm", curves, normals)
    signed = curves_proj[span_indices] - points_proj[:, np.newaxis]

    # find the two curve points closest to the plane for each point
    closest = np.argsort(np.abs(signed), axis=1)[:, :2]
    closest.sort(axis=1)  # keep the order of the curve, not the argsort order
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
) -> list[PointDistanceResult]:
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
        PointDistanceResult(
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
    sample_vertex: np.ndarray

    def points(self, seg_ids: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Points at parameters `t` of segments `seg_ids`, shape ``(N, 3)``."""
        return (
            self.seg_start[seg_ids] + t[:, np.newaxis] * self.seg_vec[seg_ids]
        )


class _CoarseSamples(NamedTuple):
    best: _BestSamples
    points: np.ndarray
    distances: np.ndarray
    targets: np.ndarray


def _build_segments_samples(
    coords_dict: dict[str, np.ndarray],
    spans_dict: dict[str, int],
    step: float,
    geometry_types: dict[str, Literal["points", "polyline"]] | None = None,
) -> _SegmentSamples:
    """Split the obstacles into segments and sample them.

    Each polyline obstacle with `V_i` vertices yields `V_i - 1` segments.
    A discrete-points obstacle (or a single-vertex one) yields one
    zero-length segment per vertex, sampled at its vertex only. Each segment
    is subdivided into `seg_sub` pieces of at most `step` meters, and
    samples are placed at local parameters `t` in `[0, 1]`, excluding the
    end vertex of every segment except the last one of its obstacle.

    Args:
        coords_dict: Obstacle name to vertices, each of shape `(V_i, 3)`.
            Its order defines the obstacle ids.
        spans_dict: Obstacle name to its span index.
        step: Maximum sampling step along the obstacle segments, in meters.
        geometry_types: Obstacle name to ``"points"`` or ``"polyline"``.
            Missing obstacles are polylines. Defaults to `None`.

    Returns:
        A `_SegmentSamples` with, for `K` segments:
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
            - `sample_vertex`: index of the obstacle vertex at the sample,
              or ``-1`` between vertices, `(P,)`.
        and `spans`: span index of each of the `N` obstacles, as a list.
    """
    geometry_types = geometry_types or {}
    seg_starts: list[np.ndarray] = []
    seg_vectors: list[np.ndarray] = []
    seg_obstacle: list[np.ndarray] = []
    seg_local: list[np.ndarray] = []
    seg_nb_sub: list[np.ndarray] = []
    sample_segments: list[np.ndarray] = []
    sample_ts: list[np.ndarray] = []
    sample_vertices: list[np.ndarray] = []
    spans: list[int] = []

    n_segments = 0
    for obstacle_id, name in enumerate(coords_dict):
        vertices = np.asarray(coords_dict[name], dtype=float)
        spans.append(int(spans_dict[name]))
        discrete = (
            len(vertices) == 1
            or geometry_types.get(name, "polyline") == "points"
        )
        if discrete:
            starts, vectors = vertices, np.zeros_like(vertices)
        else:
            starts, vectors = vertices[:-1], np.diff(vertices, axis=0)
        nb_segments = len(starts)
        nb_subdivisions = np.maximum(
            np.ceil(np.linalg.norm(vectors, axis=1) / step), 1
        ).astype(int)
        for local in range(nb_segments):
            ts = np.linspace(0.0, 1.0, nb_subdivisions[local] + 1)
            if discrete:
                ts = ts[:1]
            elif local < nb_segments - 1:
                ts = ts[:-1]
            vertex = np.full(len(ts), -1)
            vertex[0] = local
            if not discrete and local == nb_segments - 1:
                vertex[-1] = local + 1
            sample_segments.append(np.full(len(ts), n_segments + local))
            sample_ts.append(ts)
            sample_vertices.append(vertex)
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
        np.concatenate(sample_vertices),
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
        # points to the caller of PositionEngine's public method:
        # caller > public method > _compute_distances > distances
        # > _coarse_sampling > _warning_no_intersection
        warnings.warn(message, NoIntersectionPlaneWarning, stacklevel=6)


def _segment_plane_distances(
    segments: _SegmentSamples,
    seg_ids: np.ndarray,
    t: np.ndarray,
    curves: np.ndarray,
    normals: np.ndarray,
    curves_proj: np.ndarray | None = None,
) -> np.ndarray:
    """Plane distances of points lying on segments.

    Args:
        segments: Segment and sample description of the obstacles.
        seg_ids: Segment index of each point, shape ``(N,)``.
        t: Local segment parameter of each point, shape ``(N,)``.
        curves: Section-frame curve points of every span, shape (S, M, 3).
        normals: Unit horizontal plane normals of every span, shape (S, 3).
        curves_proj: Optional cache of the projections of `curves`, shape (S, M).

    Returns:
        Distances of the points, shape ``(N,)``.
    """
    distances, _ = batch_plane_distances(
        segments.points(seg_ids, t),
        segments.seg_span[seg_ids],
        curves,
        normals,
        curves_proj=curves_proj,
    )
    return distances


def _coarse_sampling(
    names: list[str],
    segments: _SegmentSamples,
    curves: np.ndarray,
    normals: np.ndarray,
    curves_proj: np.ndarray | None = None,
) -> _CoarseSamples:
    """Evaluate every sample and keep the closest one of each obstacle.

    Obstacles having samples without intersection are reported with a warning.
    On ties, the first sample is kept.

    Returns:
        The best sample of each obstacle, and the points, distances and
        targets of every sample.
    """
    sample_obs = segments.sample_obs
    sample_points = segments.points(segments.sample_seg, segments.sample_t)
    sample_distances, sample_targets = batch_plane_distances(
        sample_points,
        segments.seg_span[segments.sample_seg],
        curves,
        normals,
        curves_proj=curves_proj,
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
    return _CoarseSamples(
        best=_BestSamples(
            seg=segments.sample_seg[best],
            t=segments.sample_t[best],
            distance=sample_distances[best],
        ),
        points=sample_points,
        distances=sample_distances,
        targets=sample_targets,
    )


def _vertex_results(
    segments: _SegmentSamples,
    coarse: _CoarseSamples,
    n_obstacles: int,
    u: np.ndarray,
    v: np.ndarray,
) -> list[dict[int, PointDistanceResult]]:
    """Distance results of the obstacle vertices, from the coarse samples.

    Returns:
        For each obstacle id, vertex index to its `DistanceResult`. Vertices
        without intersection are absent.
    """
    rows = np.flatnonzero(
        (segments.sample_vertex >= 0) & np.isfinite(coarse.distances)
    )
    distance_results = build_distance_results(
        coarse.points[rows],
        segments.seg_span[segments.sample_seg[rows]],
        coarse.targets[rows],
        coarse.distances[rows],
        u,
        v,
    )
    vertices: list[dict[int, PointDistanceResult]] = [
        {} for _ in range(n_obstacles)
    ]
    for row, distance_result in zip(rows.tolist(), distance_results):
        vertices[segments.sample_obs[row]][
            int(segments.sample_vertex[row])
        ] = distance_result
    return vertices


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
    distance_results: list[PointDistanceResult],
) -> dict[str, ClosestDistanceResult | None]:
    """Map the found distance results back to their obstacle names.

    `found` holds the obstacle indices of `distance_results`, in the same
    order. Obstacles absent from `found` keep `None`.
    """
    results: dict[str, ClosestDistanceResult | None] = {
        name: None for name in names
    }
    for obstacle_id, distance_result in zip(found.tolist(), distance_results):
        results[names[obstacle_id]] = ClosestDistanceResult(
            obstacle_name=names[obstacle_id],
            span_index=spans[obstacle_id],
            distance_result=distance_result,
            segment_index=int(seg_loc[best_samples.seg[obstacle_id]]),
            segment_parameter=float(best_samples.t[obstacle_id]),
        )
    return results


def _previous_nonzero_segment(
    seg_len: np.ndarray, segment_id: int, first_seg: int
) -> int | None:
    """Closest segment before `segment_id` with positive length, within bounds."""
    for candidate in range(segment_id - 1, first_seg - 1, -1):
        if seg_len[candidate] > 0.0:
            return candidate
    return None


def _next_nonzero_segment(
    seg_len: np.ndarray, segment_id: int, last_seg: int
) -> int | None:
    """Closest segment after `segment_id` with positive length, within bounds."""
    for candidate in range(segment_id + 1, last_seg + 1):
        if seg_len[candidate] > 0.0:
            return candidate
    return None


def search_interval_refinement(
    best_samples: _BestSamples,
    seg_nb_vertices: np.ndarray,
    seg_sub: np.ndarray,
    seg_loc: np.ndarray,
    seg_len: np.ndarray,
) -> list[_SearchInterval]:
    """Build the golden-section search intervals around each coarse minimum.

    Segments of an obstacle are never crossed outside its own bounds, and
    zero-length segments (repeated vertices) never get an interval.

    - Best segment with positive length: an interval around the coarse
      minimum, plus the end of the previous nonzero segment if the minimum
      sits on the segment start (shared vertex).
    - Best segment with zero length: the start of the next nonzero segment
      and the end of the previous nonzero segment.

    Returns:
        The intervals, ordered per obstacle: current, next, then previous.
    """
    intervals: list[_SearchInterval] = []
    obstacle_id_to_refine = np.flatnonzero(
        np.isfinite(best_samples.distance) & (seg_nb_vertices > 1)
    )  # refine only obstacles with finite distance and more than one segment vertex
    for obstacle_id in map(int, obstacle_id_to_refine):
        segment_id = int(best_samples.seg[obstacle_id])
        t0 = float(best_samples.t[obstacle_id])
        first_seg = segment_id - int(seg_loc[segment_id])
        last_seg = first_seg + int(seg_nb_vertices[obstacle_id]) - 2

        on_segment = seg_len[segment_id] > 0.0
        if on_segment:
            dt = 1.0 / seg_sub[segment_id]  # in parameter unit
            intervals.append(
                _SearchInterval(
                    obstacle_id,
                    segment_id,
                    max(t0 - dt, 0.0),
                    min(t0 + dt, 1.0),
                )
            )
        else:
            # repeated vertex: search the start of the next nonzero segment
            next_seg = _next_nonzero_segment(seg_len, segment_id, last_seg)
            if next_seg is not None:
                intervals.append(
                    _SearchInterval(
                        obstacle_id, next_seg, 0.0, 1.0 / seg_sub[next_seg]
                    )
                )
        # the minimum may sit on the vertex shared with the previous nonzero segment
        if not on_segment or t0 <= 0.0:
            prev_seg = _previous_nonzero_segment(
                seg_len, segment_id, first_seg
            )
            if prev_seg is not None:
                intervals.append(
                    _SearchInterval(
                        obstacle_id,
                        prev_seg,
                        1.0 - 1.0 / seg_sub[prev_seg],
                        1.0,
                    )
                )

    return intervals
