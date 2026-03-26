# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from mechaphlowers.core.geometry.distances import (
    DistanceEngine,
    DistanceResult,
)
from mechaphlowers.core.geometry.planes import change_local_frame
from mechaphlowers.core.geometry.points import Points, SectionPoints
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
        self.section_pts = SectionPoints(
            section_array=self.section_array,
            span_model=self.span_model,
            cable_loads=self.cable_loads,
            get_displacement=balance_engine.get_displacement,
        )

    def reset(self, balance_engine: BalanceEngine) -> None:
        """Reset geometry state from `balance_engine`.

        Called automatically by `update` when the balance engine
        notifies; can also be called manually after direct modifications to
        the section array.

        Raises:
            TypeError: If `balance_engine` is not a [BalanceEngine][].
        """
        if not isinstance(balance_engine, BalanceEngine):
            raise TypeError(
                "balance_engine must be an instance of BalanceEngine"
            )
        if balance_engine.initialized is False:
            self.initialize_engine(balance_engine)
        self.section_pts.reset()

    def update(self, notifier: Notifier) -> None:
        """Observer callback — invoked by `BalanceEngine` on state change."""
        logger.debug("Position engine notified from balance engine.")
        if isinstance(notifier, BalanceEngine):
            self.reset(balance_engine=notifier)
            # Propagate the notification to any downstream observers (e.g. PlotEngine).
            self.notify()

    # ── Obstacle management ───────────────────────────────────────────────────

    def add_obstacles(self, obstacles_array: ObstacleArray) -> None:
        """Attach an `ObstacleArray` for coordinate computation."""
        self.obstacles_array = obstacles_array
        self.section_pts.add_obstacles(obstacles_array)

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
        return self.section_pts.get_spans(frame).points(True)

    def get_supports_points(self) -> np.ndarray:
        """Return support structure points (absolute section frame)."""
        return self.section_pts.get_supports().points(True)

    def get_insulators_points(self) -> np.ndarray:
        """Return insulator attachment points (absolute section frame)."""
        return self.section_pts.get_insulators().points(True)

    def get_obstacles_points(self) -> np.ndarray:
        """Return obstacle coordinates transformed to the section frame."""
        return self.section_pts.compute_obstacle_coords().points(True)

    def obstacles_dict(self) -> dict:
        """Return obstacle coordinates keyed by obstacle name.

        Format: ``{'obs_0': [[x0, y0, z0], [x1, y1, z1], ...]}``.
        """
        return self.section_pts.obstacles_dict()

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
        return self.section_pts.get_points_for_plot(project, frame_index)

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

        ground_supports = self.section_pts.supports_ground_coords
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

        ground_supports = self.section_pts.supports_ground_coords.copy()
        if span_index < 0 or span_index >= len(ground_supports) - 1:
            raise IndexError(
                f"span_index {span_index} out of range"
                f" [0, {len(ground_supports) - 2}]"
            )

        self.distance_engine.add_span_frame(
            ground_supports[span_index], ground_supports[span_index + 1]
        )
        self.distance_engine.add_curves(
            self.section_pts.get_spans(frame="section").coords[span_index]
        )
        return self.distance_engine.plane_distance(point, frame="span")

    # ── String representations ────────────────────────────────────────────────

    def __str__(self) -> str:
        return (
            f"number of supports: {self.section_array.data.span_length.shape[0]}\n"
            f"parameter: {self.span_model.sagging_parameter}\n"
            f"wind: {self.cable_loads.wind_pressure}\n"
            f"ice: {self.cable_loads.ice_thickness}\n"
            f"beta: {self.beta}\n"
        )

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}\n{self.__str__()}"
