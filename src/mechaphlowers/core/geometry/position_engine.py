# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np

from mechaphlowers.core.geometry.distances import (
    DistanceEngine,
    DistanceResult,
    ObstacleDistanceResult,
    batch_plane_distances,
    plane_basis,
)
from mechaphlowers.core.geometry.group_points import GroupPoints
from mechaphlowers.core.geometry.planes import change_local_frame
from mechaphlowers.core.geometry.points import CoordsCalculator, Points
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.entities.arrays import ObstacleArray
from mechaphlowers.entities.errors import NoIntersectionPlaneWarning
from mechaphlowers.entities.reactivity import Notifier, Observer

logger = logging.getLogger(__name__)


def _validate_sampling(step: float, xtol: float) -> None:
    if step <= 0:
        raise ValueError("step must be strictly positive")
    if xtol <= 0:
        raise ValueError("xtol must be strictly positive")


def _golden_section_search(
    evaluate: Callable[[np.ndarray], np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    tol: np.ndarray,
) -> np.ndarray:
    """Golden-section search of several independent intervals at once.

    `evaluate` takes one abscissa per interval and returns the matching
    distances. All intervals share the same number of iterations, computed
    from the largest ratio ``(upper - lower) / tol``.

    Returns:
        The midpoint of each final interval, shape of `lower`.
    """
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    ratio = np.max((upper - lower) / tol)
    n_iterations = int(np.ceil(np.log(max(ratio, 1.0)) / -np.log(inv_phi)))

    a, b = lower.copy(), upper.copy()
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = evaluate(c), evaluate(d)
    for _ in range(n_iterations):
        left = fc < fd
        a, b = np.where(left, a, c), np.where(left, d, b)
        probe = np.where(left, b - inv_phi * (b - a), a + inv_phi * (b - a))
        f_probe = evaluate(probe)
        c, d = np.where(left, probe, d), np.where(left, c, probe)
        fc, fd = np.where(left, f_probe, fd), np.where(left, fc, f_probe)
    return (a + b) / 2.0


class PositionEngine(Observer, Notifier):
    """PositionEngine computes point positions, distances, and coordinates.

    Observes a `BalanceEngine` and updates its internal geometry state
    whenever the balance engine notifies observers.  It is also a
    `Notifier` itself, so downstream observers (e.g. a
    `PlotEngine`) are automatically
    notified after every update.

    Users can work with a `PositionEngine` directly — without any Plotly
    dependency — to obtain span points, support positions, obstacle
    coordinates, and point-to-cable distances.

    Args:
        balance_engine: `BalanceEngine` to observe.

    Examples:
        >>> from mechaphlowers.core.geometry.position_engine import PositionEngine
        >>> pos_engine = PositionEngine(balance_engine)
        >>> pos_engine.get_supports_points()
        array(...)
        >>> pos_engine.get_spans_points(frame="section")
        array(...)
    """

    def __init__(self, balance_engine: BalanceEngine) -> None:
        Notifier.__init__(self)
        balance_engine.bind_to(self)

        self.distance_engine = DistanceEngine()
        self.initialize_engine(balance_engine)
        self.reset(balance_engine=balance_engine)

    def initialize_engine(self, balance_engine: BalanceEngine) -> None:
        """Initialise internal references from `balance_engine`."""
        self.span_model = balance_engine.balance_model.nodes_span_model
        self.cable_loads = balance_engine.cable_loads
        self.section_array = balance_engine.section_array
        self.virtual_supports_list = balance_engine.virtual_supports_list
        self.obstacle_array = ObstacleArray.build_empty_array()
        self.additional_points_array = ObstacleArray.build_empty_array()
        self.coords_calculator = CoordsCalculator(
            section_array=self.section_array,
            span_model=self.span_model,
            cable_loads=self.cable_loads,
            get_displacement=balance_engine.get_displacement,
            obstacle_array=self.obstacle_array,
            additional_points_array=self.additional_points_array,
        )

    @property
    def section_pts(self):
        warnings.warn(
            "section_pts was renamed coords_calculator. Use self.coords_calculator instead",
            DeprecationWarning,
        )
        return self.coords_calculator

    def reset(self, balance_engine: BalanceEngine) -> None:
        """Reset geometry state from `balance_engine`.

        Called automatically by `update` when the balance engine
        notifies; can also be called manually after direct modifications to
        the section array.

        Raises:
            TypeError: If `balance_engine` is not a `BalanceEngine`.
        """
        if not isinstance(balance_engine, BalanceEngine):
            raise TypeError(
                "balance_engine must be an instance of BalanceEngine"
            )
        if balance_engine.initialized is False:
            self.initialize_engine(balance_engine)
        self.coords_calculator.reset()

    def update(self, notifier: Notifier) -> None:
        """Observer callback — invoked by `BalanceEngine` on state change."""
        logger.debug("Position engine notified from balance engine.")
        if isinstance(notifier, BalanceEngine):
            self.reset(balance_engine=notifier)
            # Propagate the notification to any downstream observers (e.g. PlotEngine).
            self.notify()

    # ── Obstacle management ───────────────────────────────────────────────────

    def add_obstacle_array(self, obstacle_array: ObstacleArray) -> None:
        """Attach an `ObstacleArray` for coordinate computation."""
        self.obstacle_array = obstacle_array
        self.coords_calculator.obstacle_array = self.obstacle_array
        self.coords_calculator.refresh_obstacles()

    def add_obstacle(
        self,
        name: str,
        span_index: int,
        coords: np.ndarray,
        object_type: str = "ground",
        support_reference: Literal['left', 'right'] = 'left',
        span_length: np.ndarray | None = None,
    ):
        """Delegate to [`ObstacleArray.add_obstacle`][mechaphlowers.entities.arrays.ObstacleArray.add_obstacle]."""
        self.obstacle_array.add_obstacle(
            name,
            span_index,
            coords,
            object_type,
            support_reference,
            span_length,
        )
        # Replace by a Notifier/Observer pattern?
        self.coords_calculator.refresh_obstacles()

    def delete_obstacle(self, obs_names_to_delete: str | list[str]) -> None:
        """Delegate to [`ObstacleArray.delete_obstacle`][mechaphlowers.entities.arrays.ObstacleArray.delete_obstacle]."""
        self.obstacle_array.delete_obstacle(obs_names_to_delete)
        self.coords_calculator.refresh_obstacles()

    def delete_point(self, obs_name: str, point_index: int) -> None:
        """Delegate to [`ObstacleArray.delete_point`][mechaphlowers.entities.arrays.ObstacleArray.delete_point]."""
        self.obstacle_array.delete_point(obs_name, point_index)
        self.coords_calculator.refresh_obstacles()

    def add_additional_points_array(
        self, additional_points_array: ObstacleArray
    ) -> None:
        """Attach an `ObstacleArray` for additional points coordinate computation."""
        self.additional_points_array = additional_points_array
        self.coords_calculator.additional_points_array = (
            self.additional_points_array
        )
        self.coords_calculator.refresh_additional_points()

    def add_additional_point(
        self,
        name: str,
        span_index: int,
        coords: np.ndarray,
        support_reference: Literal['left', 'right'] = 'left',
        span_length: np.ndarray | None = None,
    ):
        """Delegate to [`ObstacleArray.add_obstacle`][mechaphlowers.entities.arrays.ObstacleArray.add_obstacle]."""
        self.additional_points_array.add_obstacle(
            name,
            span_index,
            coords,
            object_type="additional",
            support_reference=support_reference,
            span_length=span_length,
        )
        self.coords_calculator.refresh_additional_points()

    def delete_additional_point(
        self, names_to_delete: str | list[str]
    ) -> None:
        """Delegate to [`ObstacleArray.delete_obstacle`][mechaphlowers.entities.arrays.ObstacleArray.delete_obstacle]."""
        self.additional_points_array.delete_obstacle(names_to_delete)
        self.coords_calculator.refresh_additional_points()

    def delete_additional_point_by_index(
        self, name: str, point_index: int
    ) -> None:
        """Delegate to [`ObstacleArray.delete_point`][mechaphlowers.entities.arrays.ObstacleArray.delete_point]."""
        self.additional_points_array.delete_point(name, point_index)
        self.coords_calculator.refresh_additional_points()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def beta(self) -> np.ndarray:
        """Load angle ($\\beta$) for each span, in radians."""
        return self.cable_loads.load_angle

    # ── Data retrieval methods ────────────────────────────────────────────────

    def get_spans_points(
        self, frame: Literal["section", "localsection", "cable"]
    ) -> np.ndarray:
        """Return span cable points in the requested coordinate frame.

        Args:
            frame: One of ``"section"``, ``"localsection"``, or ``"cable"``.

        Returns:
            numpy array of shape ``(n_points, 3)``.
        """
        return self.coords_calculator.get_spans(frame).points(True)

    def get_supports_points(self) -> np.ndarray:
        """Return support structure points (absolute section frame)."""
        return self.coords_calculator.get_supports().points(True)

    def get_insulators_points(self) -> np.ndarray:
        """Return insulator attachment points (absolute section frame)."""
        return self.coords_calculator.get_insulators().points(True)

    def get_obstacles_points(self) -> np.ndarray:
        """Return obstacle coordinates transformed to the section frame."""
        return self.coords_calculator.compute_obstacle_coords().points(True)

    def get_additional_points(self) -> np.ndarray:
        """Return additional points coordinates transformed to the section frame."""
        return (
            self.coords_calculator.compute_additional_points_coords().points(
                True
            )
        )

    def obstacles_dict(self, project=False, frame_index=0) -> dict:
        """Return obstacle coordinates keyed by obstacle name.

        Format: ``{'obs_0': [[x0, y0, z0], [x1, y1, z1], ...]}``.
        """
        return self.coords_calculator.obstacles_dict(project, frame_index)

    def get_loads_coords(
        self, project: bool = False, frame_index: int = 0
    ) -> dict:
        """Return a dictionary of load coordinates indexed by span.

        If loads exist on spans 0 and 2, the result looks like:
        ``{0: [x0, y0, z0], 2: [x2, y2, z2]}``.

        Args:
            project: ``True`` to project all objects into a support frame
                (for 2-D graphs). Defaults to ``False``.
            frame_index: Index of the support frame used for projection.
                Must be in ``[0, nb_supports - 1]``.  Unused when
                `project` is `False`.  Defaults to ``0``.

        Returns:
            Dict mapping span index (``int``) to coordinate array of shape
            ``(3,)``.
        """
        spans_points, _, _ = self.get_points_for_plot(project, frame_index)
        loads_spans_idx, loads_points_idx = self.span_model.loads_indices
        result_dict: dict = {}
        for index_in_small_array, span_index in enumerate(loads_spans_idx):
            point_index = loads_points_idx[index_in_small_array]
            result_dict[int(span_index)] = spans_points.coords[
                span_index, point_index
            ]
        return result_dict

    def get_loads_coords_group_points(
        self, project: bool = False, frame_index: int = 0
    ) -> dict:
        """Same as get_loads_coords() but uses GroupPoints object

        Return a dictionary of load coordinates indexed by span.

        If loads exist on spans 0 and 2, the result looks like:
        ``{0: [x0, y0, z0], 2: [x2, y2, z2]}``.

        Args:
            project: ``True`` to project all objects into a support frame
                (for 2-D graphs). Defaults to ``False``.
            frame_index: Index of the support frame used for projection.
                Must be in ``[0, nb_supports - 1]``.  Unused when
                `project` is `False`.  Defaults to ``0``.

        Returns:
            Dict mapping span index (``int``) to coordinate array of shape
            ``(3,)``.
        """
        group_points = self.get_group_points()
        if project:
            group_points = group_points.change_frame(frame_index)
        spans_points = group_points.spans
        loads_spans_idx, loads_points_idx = self.span_model.loads_indices
        result_dict: dict = {}
        for index_in_small_array, span_index in enumerate(loads_spans_idx):
            point_index = loads_points_idx[index_in_small_array]
            result_dict[int(span_index)] = spans_points.coords[  # type: ignore[union-attr]
                span_index, point_index
            ]
        return result_dict

    def get_points_for_plot(
        self, project: bool = False, frame_index: int = 0
    ) -> tuple[Points, Points, Points]:
        """Return `Points` objects for spans, supports, and insulators.

        Args:
            project: `True` to project into a support frame (2-D mode).
            frame_index: Index of the support frame for projection.

        Returns:
            Tuple of ``(spans, supports, insulators)`` as `Points`.

        Raises:
            ValueError: If `frame_index` is out of range.
        """
        return self.coords_calculator.get_points_for_plot(project, frame_index)

    # ── Coordinate / distance analysis ────────────────────────────────────────

    def point_relative_to_absolute(
        self, span_index: int, point_relative: np.ndarray
    ) -> np.ndarray:
        """Convert a point from the span-local frame to absolute coordinates.

        Span-local frame definition:

        - X: along the span direction projected onto the XY plane
        - Y: perpendicular to X in the XY plane
        - Z: vertical (global Z)

        Args:
            span_index: Span index in ``[0, num_supports - 2]``.
            point_relative: Coordinate ``[x, y, z]`` in the span-local frame.

        Returns:
            Absolute coordinate array of shape ``(3,)``.

        Raises:
            IndexError: If `span_index` is out of range.
            ValueError: If `point_relative` does not have shape ``(3,)``.
        """
        point_relative = np.asarray(point_relative)
        if point_relative.shape != (3,):
            raise ValueError("point_relative must be a 1D array of shape (3,)")

        ground_supports = self.coords_calculator.supports_ground_coords
        if span_index < 0 or span_index >= len(ground_supports) - 1:
            raise IndexError(
                f"span_index {span_index} out of range"
                f" [0, {len(ground_supports) - 2}]"
            )

        return change_local_frame(
            ground_supports[span_index],
            ground_supports[span_index + 1],
            point_relative,
        )

    def point_distance(
        self, span_index: int, point: np.ndarray
    ) -> DistanceResult:
        """Compute the minimum distance from `point` to a cable span.

        Args:
            span_index: Span index in ``[0, num_supports - 2]``.
            point: Absolute coordinates of shape ``(3,)``.

        Returns:
            `DistanceResult` with the distance value and closest-point coordinates.

        Raises:
            IndexError: If `span_index` is out of range.
            ValueError: If `point` does not have shape ``(3,)``.
        """
        point = np.asarray(point)
        if point.shape != (3,):
            raise ValueError("point must be a 1D array of shape (3,)")

        self._set_distance_span(span_index)
        return self.distance_engine.plane_distance(point, frame="section")

    def _set_distance_span(
        self, span_index: int, spans_coords: np.ndarray | None = None
    ) -> None:
        """Load span `span_index` frame and cable curve into the distance engine.

        `spans_coords` is the section-frame spans coordinates, if already computed.
        """
        ground_supports = self.coords_calculator.supports_ground_coords.copy()
        if span_index < 0 or span_index >= len(ground_supports) - 1:
            raise IndexError(
                f"span_index {span_index} out of range"
                f" [0, {len(ground_supports) - 2}]"
            )

        if spans_coords is None:
            spans_coords = self.coords_calculator.get_spans(
                frame="section"
            ).coords
        self.distance_engine.add_span_frame(
            ground_supports[span_index], ground_supports[span_index + 1]
        )
        self.distance_engine.add_curves(spans_coords[span_index])

    def obstacle_min_distance(
        self,
        obstacle_name: str,
        step: float = 1.0,
        refine: bool = True,
        xtol: float = 1e-2,
    ) -> ObstacleDistanceResult | None:
        """Compute the minimum distance between a polyline obstacle and its span.

        The obstacle points are joined, in `point_index` order, into a
        polyline. Each segment is sampled every `step` meters (vertices
        included) and the plane distance (see
        [`point_distance`][mechaphlowers.core.geometry.position_engine.PositionEngine.point_distance])
        is computed at each sample. When `refine` is `True`, a golden-section
        search is then run around the best sample to find the minimum
        between samples.

        Samples whose distance plane does not intersect the span (e.g. beyond
        the supports) are skipped and a `NoIntersectionPlaneWarning` is
        emitted.

        Args:
            obstacle_name: Name of the obstacle.
            step: Sampling step along the obstacle segments, in meters.
                Defaults to ``1.0``.
            refine: Refine the minimum between samples. Defaults to `True`.
            xtol: Absolute tolerance of the refinement, in meters along the
                segment. Defaults to ``1e-3``.

        Returns:
            [`ObstacleDistanceResult`][mechaphlowers.core.geometry.distances.ObstacleDistanceResult],
            or `None` if no obstacle sample intersects the span.

        Raises:
            KeyError: If `obstacle_name` does not exist.
            ValueError: If `step` or `xtol` is not strictly positive.

        Examples:
            >>> pos_engine.add_obstacle(
            ...     name="road",
            ...     span_index=0,
            ...     coords=np.array([[100, -20, 0], [150, 20, 2]]),
            ... )
            >>> result = pos_engine.obstacle_min_distance("road")
            >>> result.distance_3d, result.closest_obstacle_point
        """
        _validate_sampling(step, xtol)
        result = self._min_distances(
            [obstacle_name], step=step, refine=refine, xtol=xtol
        )[obstacle_name]
        if result is not None:
            # Load the distance engine so that plotting uses the same plane.
            self._set_distance_span(result.span_index)
            self.distance_engine.plane_distance(
                result.closest_obstacle_point, frame="section"
            )
        return result

    def _span_geometry(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Helper to get section-frame curves, horizontal normals and plane basis of all spans.

        Returns:
            A tuple `(curves, normals, u, v)` with shapes (S, M, 3), (S, 3),
            (S, 3) and (S, 3).
        """
        ground = np.array(
            self.coords_calculator.supports_ground_coords, dtype=float
        )
        # project ground points onto the XY plane
        ground[:, 2] = 0.0
        # get the direction vectors of the spans in the XY plane
        axis = np.diff(ground, axis=0)
        # normalize the direction vectors to get horizontal normals
        normals = axis / np.linalg.norm(axis, axis=1, keepdims=True)

        # get the section-frame curves for all spans and u,v basis for the planes
        curves = self.coords_calculator.get_spans(frame="section").coords
        u, v = plane_basis(normals)
        return curves, normals, u, v

    def _build_distance_results(
        self,
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

    def _min_distances(
        self,
        names: list[str] | None,
        step: float,
        refine: bool,
        xtol: float,
    ) -> dict[str, ObstacleDistanceResult | None]:
        """Minimum distance of the selected obstacles, computed together.

        All samples of all obstacles are evaluated in one batch. When `refine`
        is `True`, the minimum around each obstacle best sample is refined by
        a golden-section search, run simultaneously for all obstacles.

        Shapes used internally, with `N` selected obstacles, `S` spans,
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
            names: Obstacle names, or `None` for all obstacles.
            step: Sampling step along the obstacle segments, in meters.
            refine: Refine the minimum between samples.
            xtol: Absolute tolerance of the refinement, in meters.

        Returns:
            Dict mapping each of the `N` obstacle names to its
            `ObstacleDistanceResult`, or `None` if no sample of the obstacle
            faces the span.

        Raises:
            KeyError: If an obstacle name does not exist.
        """

        # get the obstacles points
        obstacles_points = self.coords_calculator.compute_obstacle_coords()
        coords_dict = obstacles_points.dict_coords()
        if names is None:
            names = list(coords_dict)
        for name in names:
            if name not in coords_dict:
                raise KeyError(f"Obstacle '{name}' not found")
        if not names:
            return {}

        object_names = list(obstacles_points.object_name)
        span_of_row = np.asarray(obstacles_points.span_index, dtype=int)
        first_row: dict[str, int] = {}
        for row, name in enumerate(object_names):
            first_row.setdefault(name, row)

        # Segments of all obstacles, and samples on each segment.
        seg_starts: list[np.ndarray] = [] # list of segment start points for all obstacles
        seg_vectors: list[np.ndarray] = [] # list of segment vectors for all obstacles
        seg_obstacle: list[np.ndarray] = [] # list of obstacle ids for each segment
        seg_local: list[np.ndarray] = [] # list of local segment indices within each obstacle
        seg_nb_sub: list[np.ndarray] = [] # list of number of subdivisions for each segment
        sample_segments: list[np.ndarray] = [] # list of segment indices for each sample
        sample_ts: list[np.ndarray] = [] # list of local t values for each sample
        spans: list[int] = [] # list of span indices for each obstacle

        # iterate over each obstacle and compute its segments and samples
        n_segments = 0
        for obstacle_id, name in enumerate(names):
            vertices = np.asarray(coords_dict[name], dtype=float) # shape (M,3) with M the number of vertices for this obstacle
            spans.append(int(span_of_row[first_row[name]]))
            if len(vertices) == 1:
                # if single obstacle, no vectors, just a single point
                starts, vectors = vertices, np.zeros_like(vertices)
            else:
                starts, vectors = vertices[:-1], np.diff(vertices, axis=0)
            nb_segments = len(starts)
            nb_subdivisions = np.maximum( # max(value, 1) allow to divide at least by one (case single point)
                np.ceil(np.linalg.norm(vectors, axis=1) / step), 1 # divide segment length by step to get number of subdivisions
            ).astype(int)
            for local in range(nb_segments):
                ts = np.linspace(0.0, 1.0, nb_subdivisions[local] + 1)
                if len(vertices) == 1:
                    ts = ts[:1]
                elif local < nb_segments - 1:
                    # Shared vertices are sampled once, as t=0 of the next segment.
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

        curves, normals, u, v = self._span_geometry()

        def evaluate(seg_ids: np.ndarray, t: np.ndarray) -> np.ndarray:
            
            points = seg_start[seg_ids] + t[:, np.newaxis] * seg_vec[seg_ids]
            distances, _ = batch_plane_distances(
                points, seg_span[seg_ids], curves, normals
            )
            return distances

        sample_distances = evaluate(sample_seg, sample_t)
        n_obstacles = len(names)
        n_samples = np.bincount(sample_obs, minlength=n_obstacles)
        n_skipped = np.bincount(
            sample_obs[~np.isfinite(sample_distances)], minlength=n_obstacles
        )
        for obstacle_id in map(int, np.flatnonzero(n_skipped)):
            message = (
                f"No intersection found between span {spans[obstacle_id]} and "
                f"{n_skipped[obstacle_id]}/{n_samples[obstacle_id]} sampled points of obstacle "
                f"'{names[obstacle_id]}' - skipping them."
            )
            logger.warning(message)
            warnings.warn(message, NoIntersectionPlaneWarning, stacklevel=3)

        # Best sample per obstacle (first one on ties).
        order = np.lexsort((sample_distances, sample_obs))
        group_start = np.searchsorted(
            sample_obs[order], np.arange(n_obstacles)
        )
        best = order[group_start]
        best_seg = sample_seg[best]
        best_t = sample_t[best]
        best_distance = sample_distances[best]

        if refine:
            seg_nb_vertices = np.array([len(coords_dict[n]) for n in names])
            intervals: list[tuple[int, int, float, float]] = []
            for obstacle_id in map(
                int,
                np.flatnonzero(
                    np.isfinite(best_distance) & (seg_nb_vertices > 1)
                ),
            ):
                segment = int(best_seg[obstacle_id])
                t0 = float(best_t[obstacle_id])
                dt = 1.0 / seg_sub[segment]
                # Search both sides of a shared vertex.
                intervals.append(
                    (
                        obstacle_id,
                        segment,
                        max(t0 - dt, 0.0),
                        min(t0 + dt, 1.0),
                    )
                )
                if t0 == 0.0 and seg_loc[segment] > 0:
                    prev_dt = 1.0 / seg_sub[segment - 1]
                    intervals.append(
                        (obstacle_id, segment - 1, 1.0 - prev_dt, 1.0)
                    )
            intervals = [iv for iv in intervals if seg_len[iv[1]] > 0.0]

            if intervals:
                obs_ids = np.array([iv[0] for iv in intervals], dtype=int)
                interval_segs = np.array(
                    [iv[1] for iv in intervals], dtype=int
                )
                lower = np.array([iv[2] for iv in intervals])
                upper = np.array([iv[3] for iv in intervals])
                refined_t = _golden_section_search(
                    lambda t: evaluate(interval_segs, t),
                    lower,
                    upper,
                    tol=xtol / seg_len[interval_segs],
                )
                refined_distance = evaluate(interval_segs, refined_t)
                for obs_id, segment, t_refined, distance in zip(
                    obs_ids, interval_segs, refined_t, refined_distance
                ):
                    if distance < best_distance[obs_id]:
                        best_seg[obs_id] = segment
                        best_t[obs_id] = t_refined
                        best_distance[obs_id] = distance

        best_points = (
            seg_start[best_seg] + best_t[:, np.newaxis] * seg_vec[best_seg]
        )
        best_spans = seg_span[best_seg]
        final_distances, final_targets = batch_plane_distances(
            best_points, best_spans, curves, normals
        )
        found = np.flatnonzero(np.isfinite(final_distances))
        distance_results = self._build_distance_results(
            best_points[found],
            best_spans[found],
            final_targets[found],
            final_distances[found],
            u,
            v,
        )

        results: dict[str, ObstacleDistanceResult | None] = {
            name: None for name in names
        }
        for obstacle_id, distance_result in zip(
            found.tolist(), distance_results
        ):
            results[names[obstacle_id]] = ObstacleDistanceResult(
                obstacle_name=names[obstacle_id],
                span_index=spans[obstacle_id],
                distance_result=distance_result,
                segment_index=int(seg_loc[best_seg[obstacle_id]]),
                segment_parameter=float(best_t[obstacle_id]),
            )
        return results

    def get_segment_distances_from_obstacles(
        self,
        step: float = 1.0,
        refine: bool = True,
        xtol: float = 1e-3,
    ) -> dict[str, ObstacleDistanceResult | None]:
        """Compute the minimum distance of every obstacle to its span.

        Each obstacle is treated as a polyline. Every segment is sampled
        every `step` metres, vertices included, and the plane distance to
        the span is computed at each sample. When `refine` is `True`, a
        golden-section search then refines the minimum around the best
        sample. If several samples share the same distance, the first one
        is kept.

        The span is represented by its sampled cable polyline
        (`cfg.graphics.resolution` points), so the distances are
        approximate and may differ slightly from the exact cable minimum.

        See [`obstacle_min_distance`][mechaphlowers.core.geometry.position_engine.PositionEngine.obstacle_min_distance]
        for the single-obstacle variant and the arguments.

        Args:
            step: Sampling step along the obstacle segments, in meters.
                Defaults to ``1.0``.
            refine: Refine the minimum between samples. Defaults to `True`.
            xtol: Absolute tolerance of the refinement, in meters.
                Defaults to ``1e-3``.

        Returns:
            Dict mapping obstacle name to its
            [`ObstacleDistanceResult`][mechaphlowers.core.geometry.distances.ObstacleDistanceResult]
            (`None` if the obstacle does not face the span).
        """
        _validate_sampling(step, xtol)
        return self._min_distances(None, step=step, refine=refine, xtol=xtol)

    def get_distances_from_obstacles(
        self,
    ) -> dict[str, dict[int, DistanceResult]]:
        """Compute distances for all obstacles to their respective spans.

        Only in absolute coordiantes.

        {
            'obs_0': {0: DistanceResult, 1: DistanceResult},
            'obs_1': {0: DistanceResult}
        }

        Returns:
            dict: dictionary of DistanceResult, sorted by obstacles
        """
        obstacles_points = self.coords_calculator.compute_obstacle_coords()
        names = list(obstacles_points.object_name)
        span_indices = np.asarray(obstacles_points.span_index, dtype=int)
        point_indices = np.asarray(obstacles_points.point_index, dtype=int)
        points = np.asarray(obstacles_points.coords, dtype=float)

        curves, normals, u, v = self._span_geometry()
        distances, targets = batch_plane_distances(
            points, span_indices, curves, normals
        )
        found = np.isfinite(distances)
        results = self._build_distance_results(
            points[found],
            span_indices[found],
            targets[found],
            distances[found],
            u,
            v,
        )

        distance_dict_result: dict[str, dict[int, DistanceResult]] = {}
        found_rows = np.flatnonzero(found)
        for name in dict.fromkeys(names):
            distance_dict_result[name] = {}
        for row, distance_result in zip(found_rows, results):
            distance_dict_result[names[row]][int(point_indices[row])] = (
                distance_result
            )

        skipped_by_name: dict[str, list[int]] = {}
        for row in np.flatnonzero(~found):
            skipped_by_name.setdefault(names[row], []).append(
                int(point_indices[row])
            )
        for name, skipped in skipped_by_name.items():
            if skipped:
                message = (
                    f"No intersection found between obstacle '{name}' "
                    f"(point index {', '.join(map(str, skipped))}) and its span - "
                    "skipping distance computation for these points."
                )
                logger.warning(message)
                warnings.warn(
                    message, NoIntersectionPlaneWarning, stacklevel=2
                )

        return distance_dict_result

    def get_group_points(self) -> GroupPoints:
        spans_points: Points = self.coords_calculator.get_spans("section")
        supports_points: Points = self.coords_calculator.get_supports()
        insulators_points: Points = self.coords_calculator.get_insulators()

        obstacles_points = None
        additional_points = None
        distances_dict = None

        if (
            hasattr(self, "obstacle_array")
            and not self.obstacle_array.data_original.empty
        ):
            self.coords_calculator.compute_obstacle_coords()
            obstacles_points = self.coords_calculator.obstacles_points

        if (
            hasattr(self, "additional_points_array")
            and not self.additional_points_array.data_original.empty
        ):
            self.coords_calculator.compute_additional_points_coords()
            additional_points = self.coords_calculator.additional_points

        # Only compute distances if we have obstacles
        if obstacles_points is not None or additional_points is not None:
            distances_dict = {}
            if obstacles_points is not None:
                distances_dict.update(self.get_distances_from_obstacles())

        return GroupPoints(
            line_angle=self.coords_calculator.line_angle,
            spans=spans_points,
            supports=supports_points,
            insulators=insulators_points,
            obstacles=obstacles_points,
            additional_points=additional_points,
            distances=distances_dict,
            virtual_supports_list=self.virtual_supports_list,
        )

    # ── String representations ────────────────────────────────────────────────

    def __str__(self) -> str:
        return (
            f"number of supports: {self.section_array.data.span_length.shape[0]}\n"
            f"parameter: {self.span_model.parameter}\n"
            f"wind: {self.cable_loads.wind_pressure}\n"
            f"ice: {self.cable_loads.ice_thickness}\n"
            f"beta: {self.beta}\n"
        )

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}\n{self.__str__()}"
