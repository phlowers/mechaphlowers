# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np

from mechaphlowers.core.geometry.distances import (
    DistanceEngine,
    DistanceResult,
    ObstacleDistanceResult,
    minimize_bounded,
)
from mechaphlowers.core.geometry.group_points import GroupPoints
from mechaphlowers.core.geometry.planes import change_local_frame
from mechaphlowers.core.geometry.points import CoordsCalculator, Points
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.entities.arrays import ObstacleArray
from mechaphlowers.entities.errors import (
    NoIntersectionPlaneForDistanceError,
    NoIntersectionPlaneWarning,
)
from mechaphlowers.entities.reactivity import Notifier, Observer

logger = logging.getLogger(__name__)


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

    def _set_distance_span(self, span_index: int) -> None:
        """Load span `span_index` frame and cable curve into the distance engine."""
        ground_supports = self.coords_calculator.supports_ground_coords.copy()
        if span_index < 0 or span_index >= len(ground_supports) - 1:
            raise IndexError(
                f"span_index {span_index} out of range"
                f" [0, {len(ground_supports) - 2}]"
            )

        self.distance_engine.add_span_frame(
            ground_supports[span_index], ground_supports[span_index + 1]
        )
        self.distance_engine.add_curves(
            self.coords_calculator.get_spans(frame="section").coords[
                span_index
            ]
        )

    def obstacle_min_distance(
        self,
        obstacle_name: str,
        step: float = 1.0,
        refine: bool = True,
        xtol: float = 1e-3,
    ) -> ObstacleDistanceResult | None:
        """Compute the minimum distance between a polyline obstacle and its span.

        The obstacle points are joined, in `point_index` order, into a
        polyline. Each segment is sampled every `step` meters (vertices
        included) and the plane distance (see
        [`point_distance`][mechaphlowers.core.geometry.position_engine.PositionEngine.point_distance])
        is computed at each sample. When `refine` is `True`, a bounded scalar
        minimization is then run around the best sample to find the minimum
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
        if step <= 0:
            raise ValueError("step must be strictly positive")
        if xtol <= 0:
            raise ValueError("xtol must be strictly positive")

        obstacles_points = self.coords_calculator.compute_obstacle_coords()
        coords_dict = obstacles_points.dict_coords()
        if obstacle_name not in coords_dict:
            raise KeyError(f"Obstacle '{obstacle_name}' not found")
        vertices = np.asarray(coords_dict[obstacle_name], dtype=float)
        span_index = int(
            obstacles_points.span_index[
                list(obstacles_points.object_name).index(obstacle_name)
            ]
        )
        self._set_distance_span(span_index)

        nb_segments = max(len(vertices) - 1, 1)
        if len(vertices) == 1:
            segment_starts = vertices
            segment_vectors = np.zeros_like(vertices)
        else:
            segment_starts = vertices[:-1]
            segment_vectors = np.diff(vertices, axis=0)
        nb_subdivisions = np.maximum(
            np.ceil(np.linalg.norm(segment_vectors, axis=1) / step), 1
        ).astype(int)

        def distance_at(segment: int, t: float) -> DistanceResult | None:
            point = segment_starts[segment] + t * segment_vectors[segment]
            try:
                return self.distance_engine.plane_distance(
                    point, frame="section"
                )
            except NoIntersectionPlaneForDistanceError:
                return None

        # Shared vertices are sampled once, as t=0 of the following segment.
        samples: list[tuple[int, float]] = []
        for segment in range(nb_segments):
            ts = np.linspace(0.0, 1.0, nb_subdivisions[segment] + 1)
            if len(vertices) == 1:
                ts = ts[:1]
            elif segment < nb_segments - 1:
                ts = ts[:-1]
            samples.extend((segment, float(t)) for t in ts)

        best: tuple[int, float, DistanceResult] | None = None
        nb_skipped = 0
        for segment, t in samples:
            result = distance_at(segment, t)
            if result is None:
                nb_skipped += 1
            elif best is None or result.distance_3d < best[2].distance_3d:
                best = (segment, t, result)

        if nb_skipped > 0:
            message = (
                f"No intersection found between span {span_index} and "
                f"{nb_skipped}/{len(samples)} sampled points of obstacle "
                f"'{obstacle_name}' - skipping them."
            )
            logger.warning(message)
            warnings.warn(message, NoIntersectionPlaneWarning, stacklevel=2)
        if best is None:
            return None

        if refine and len(vertices) > 1:
            best_segment, best_t, _ = best
            # Search the neighbourhood of the best sample, on both sides of
            # a shared vertex.
            intervals = []
            dt = 1.0 / nb_subdivisions[best_segment]
            intervals.append(
                (best_segment, max(best_t - dt, 0.0), min(best_t + dt, 1.0))
            )
            if best_t == 0.0 and best_segment > 0:
                prev_dt = 1.0 / nb_subdivisions[best_segment - 1]
                intervals.append((best_segment - 1, 1.0 - prev_dt, 1.0))

            for segment, t_min, t_max in intervals:
                segment_length = float(
                    np.linalg.norm(segment_vectors[segment])
                )
                if segment_length == 0.0:
                    continue

                def objective(t: float, segment: int = segment) -> float:
                    result = distance_at(segment, t)
                    return np.inf if result is None else result.distance_3d

                t_refined = minimize_bounded(
                    objective, (t_min, t_max), xtol=xtol / segment_length
                )
                result = distance_at(segment, t_refined)
                if (
                    result is not None
                    and result.distance_3d < best[2].distance_3d
                ):
                    best = (segment, t_refined, result)

        segment, t, best_result = best
        # Re-evaluate so that the distance engine state matches the result.
        distance_at(segment, t)
        return ObstacleDistanceResult(
            obstacle_name=obstacle_name,
            span_index=span_index,
            distance_result=best_result,
            segment_index=segment,
            segment_parameter=t,
        )

    def get_min_distances_from_obstacles(
        self,
        step: float = 1.0,
        refine: bool = True,
        xtol: float = 1e-3,
    ) -> dict[str, ObstacleDistanceResult | None]:
        """Compute the minimum distance of every obstacle to its span.

        See [`obstacle_min_distance`][mechaphlowers.core.geometry.position_engine.PositionEngine.obstacle_min_distance]
        for the algorithm and arguments.

        Returns:
            Dict mapping obstacle name to its
            [`ObstacleDistanceResult`][mechaphlowers.core.geometry.distances.ObstacleDistanceResult]
            (`None` if the obstacle does not face the span).
        """
        obstacles_points = self.coords_calculator.compute_obstacle_coords()
        return {
            obstacle_name: self.obstacle_min_distance(
                obstacle_name, step=step, refine=refine, xtol=xtol
            )
            for obstacle_name in obstacles_points.dict_coords()
        }

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

        distance_dict_result = {}
        # self.coord_calculator.compute_obstacle_coords()?
        obstacle_sparse_points = self.coords_calculator.obstacles_points
        # index of obstacle point among total points
        loop_index = 0
        for (
            obstacle_name,
            obstacle_coords_array,
        ) in obstacle_sparse_points.dict_coords().items():
            current_distance_result = {}
            # create dict {0: DistanceResult, 1:: DistanceResult, ...} per obstacle
            for obstacle_coords in obstacle_coords_array:
                span_index = obstacle_sparse_points.span_index[loop_index]
                point_index = obstacle_sparse_points.point_index[loop_index]
                try:
                    distance_result = self.point_distance(
                        span_index, obstacle_coords
                    )
                except NoIntersectionPlaneForDistanceError as error:
                    message = (
                        f"No intersection found between obstacle '{obstacle_name}' "
                        f"(point index {point_index}) and span {span_index} - "
                        f"skipping distance computation for this point. ({error})"
                    )
                    logger.warning(message)
                    warnings.warn(
                        message, NoIntersectionPlaneWarning, stacklevel=2
                    )
                    loop_index += 1
                    continue
                current_distance_result[point_index] = distance_result
                loop_index += 1
            distance_dict_result[obstacle_name] = current_distance_result

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
