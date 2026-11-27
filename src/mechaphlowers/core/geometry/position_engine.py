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
    ClosestDistanceResult,
    DistanceEngine,
    ObstacleDistances,
    PointDistanceResult,
    VectorizedDistanceEngine,
)
from mechaphlowers.core.geometry.group_points import GroupPoints
from mechaphlowers.core.geometry.planes import change_local_frame
from mechaphlowers.core.geometry.points import CoordsCalculator, Points
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.entities.arrays import ObstacleArray
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
    coordinates, and obstacle-to-cable distances (see `get_distances`).

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
        self.vectorized_distance_engine = VectorizedDistanceEngine(
            ground_supports=self.coords_calculator.supports_ground_coords,
            curves=self.coords_calculator.get_spans(frame="section").coords,
        )

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
        geometry_type: Literal['points', 'polyline'] = 'points',
    ):
        """Delegate to [`ObstacleArray.add_obstacle`][mechaphlowers.entities.arrays.ObstacleArray.add_obstacle]."""
        self.obstacle_array.add_obstacle(
            name,
            span_index,
            coords,
            object_type,
            support_reference,
            span_length,
            geometry_type=geometry_type,
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
    ) -> PointDistanceResult:
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
        return self.distance_engine.scalar_plane_distance(
            point, frame="section"
        )

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

    def get_distances(
        self,
        names: str | list[str] | None = None,
        step: float = 1.0,
        refine: bool = True,
        xtol: float = 1e-3,
    ) -> dict[str, ObstacleDistances]:
        """Compute the distances between obstacles and their span.

        For every selected obstacle:

        - each obstacle point gets its plane distance to the cable
          (see [`point_distance`][mechaphlowers.core.geometry.position_engine.PositionEngine.point_distance]),
          with the target point on the cable;
        - for ``geometry_type="points"`` obstacles (default), the minimum is
          the closest point;
        - for ``geometry_type="polyline"`` obstacles with at least two points,
          the minimum is searched along the polyline: each segment is sampled
          every `step` meters (vertices included) and, when `refine` is
          `True`, a golden-section search refines the minimum between samples.
          Otherwise, the minimum is the closest sample.

        All obstacles are computed in batched numpy operations. The span is
        represented by its sampled cable polyline
        (`cfg.graphics.resolution` points), so the distances are
        approximate.

        Points or samples whose distance plane does not intersect the span
        (e.g. beyond the supports) are skipped and a
        `NoIntersectionPlaneWarning` is emitted.

        Args:
            names: Obstacle name, list of names, or `None` for all obstacles.
            step: Sampling step along the polyline segments, in meters.
                Defaults to ``1.0``.
            refine: Refine the polyline minimum between samples.
                Defaults to `True`.
            xtol: Absolute tolerance of the refinement, in meters along the
                segment. Defaults to ``1e-3``.

        Returns:
            Dict mapping obstacle name to its
            [`ObstacleDistances`][mechaphlowers.core.geometry.distances.ObstacleDistances].

        Raises:
            KeyError: If an obstacle name does not exist.
            ValueError: If `step` or `xtol` is not strictly positive.

        Examples:
            >>> pos_engine.add_obstacle(
            ...     name="road",
            ...     span_index=0,
            ...     coords=np.array([[100, -20, 0], [150, 20, 2]]),
            ...     geometry_type="polyline",
            ... )
            >>> distances = pos_engine.get_distances("road")["road"]
            >>> distances.points[0].distance_3d  # distance of the first point
            >>> distances.minimum.distance_3d  # minimum along the polyline
        """
        return self._compute_distances(names, step, refine, xtol)

    def obstacle_min_distance(
        self,
        obstacle_name: str,
        step: float = 1.0,
        refine: bool = True,
        xtol: float = 1e-3,
    ) -> ClosestDistanceResult | None:
        """Compute the minimum distance between an obstacle and its span.

        Shortcut for the `minimum` of
        [`get_distances`][mechaphlowers.core.geometry.position_engine.PositionEngine.get_distances],
        which also loads the distance engine with the closest location, so
        that plotting uses the same plane.

        Args:
            obstacle_name: Name of the obstacle.
            step: Sampling step along the polyline segments, in meters.
                Defaults to ``1.0``.
            refine: Refine the polyline minimum between samples.
                Defaults to `True`.
            xtol: Absolute tolerance of the refinement, in meters along the
                segment. Defaults to ``1e-3``.

        Returns:
            [`ObstacleDistanceResult`][mechaphlowers.core.geometry.distances.ObstacleDistanceResult],
            or `None` if no obstacle point intersects the span.

        Raises:
            KeyError: If `obstacle_name` does not exist.
            ValueError: If `step` or `xtol` is not strictly positive.
        """
        result = self._compute_distances([obstacle_name], step, refine, xtol)[
            obstacle_name
        ].minimum
        if result is not None:
            # Load the distance engine so that plotting uses the same plane.
            self._set_distance_span(result.span_index)
            self.distance_engine.scalar_plane_distance(
                result.closest_obstacle_point, frame="section"
            )
        return result

    def update_distance_spans(self, span_ids: list[int] | None = None) -> None:
        """Refresh the spans used by `get_distances` and `obstacle_min_distance`.

        `get_distances` uses the stored span frames and curves as they are:
        it does not recompute them. Call this method after a solve or any
        change of the state (cable loads, section, supports, ...), otherwise
        the distances are computed on the previous geometry.

        Args:
            span_ids: Indices of the spans whose curves are refreshed. `None`
                refreshes the frames and the curves of all spans.

        Raises:
            IndexError: If a span index is out of range.
        """
        engine = self.vectorized_distance_engine
        if span_ids is None:
            engine.set_spans(
                self.coords_calculator.supports_ground_coords,
                self.coords_calculator.get_spans(frame="section").coords,
            )
            return
        ids = np.asarray(span_ids, dtype=int)
        curves = self.coords_calculator.get_spans(frame="section").coords
        engine.update_spans(ids, curves[ids])

    def _compute_distances(
        self,
        names: str | list[str] | None,
        step: float,
        refine: bool,
        xtol: float,
    ) -> dict[str, ObstacleDistances]:
        """Implementation of `get_distances`.

        Uses the spans stored in `vectorized_distance_engine` without
        refreshing them (see `update_distance_spans`). Must be called directly
        by the public methods: the warning stack levels rely on it.
        """
        obstacles_points = self.coords_calculator.compute_obstacle_coords()
        row_names = np.asarray(obstacles_points.object_name, dtype=object)
        all_names = list(dict.fromkeys(row_names.tolist()))
        if names is None:
            selected = all_names
        else:
            selected = list(
                dict.fromkeys([names] if isinstance(names, str) else names)
            )
            missing = set(selected).difference(all_names)
            if missing:
                raise KeyError(f"Obstacle(s) not found: {sorted(missing)}")
        if not selected:
            return {}

        geometry_types = dict(
            zip(
                self.obstacle_array.data["name"],
                self.obstacle_array.data["geometry_type"],
            )
        )
        points = np.asarray(obstacles_points.coords, dtype=float)
        span_indices = np.asarray(obstacles_points.span_index, dtype=int)

        # Rows are sorted by (name, point_index): vertices in point_index order
        return self.vectorized_distance_engine.distances(
            {name: points[row_names == name] for name in selected},
            {
                name: int(span_indices[row_names == name][0])
                for name in selected
            },
            geometry_types={name: geometry_types[name] for name in selected},
            step=step,
            refine=refine,
            xtol=xtol,
        )

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
                self.update_distance_spans()
                distances_dict.update(self.get_distances())

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
