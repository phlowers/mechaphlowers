# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal, get_args

if TYPE_CHECKING:
    from mechaphlowers.core.geometry.distances import DistanceResult

DistanceType = Literal["distance_3d", "horizontal", "vertical"]


class ComputeDistance:
    """Select the distance component used as an estimation target.

    Args:
        distance_type: One of:

            - ``"distance_3d"``: unsigned 3D distance.
            - ``"horizontal"``: signed component along the plane ``u`` axis.
            - ``"vertical"``: signed ``z_cable - z_obstacle``.

    Raises:
        ValueError: If ``distance_type`` is not a valid value.
    """

    def __init__(self, distance_type: DistanceType) -> None:
        valid_types = get_args(DistanceType)
        if distance_type not in valid_types:
            raise ValueError(
                f"distance_type must be one of {', '.join(repr(t) for t in valid_types)}, "
                f"got {distance_type!r}"
            )
        self.distance_type = distance_type

    def select_distance(self, result: DistanceResult) -> float:
        """Return the distance component selected by ``distance_type``.

        Args:
            result: Distance result between the obstacle (base point) and the
                cable (target point).

        Returns:
            The selected distance in meters. Signed for ``"horizontal"`` and
            ``"vertical"``.
        """
        if self.distance_type == "distance_3d":
            return float(result.distance_3d)
        if self.distance_type == "horizontal":
            return float(result.signed_distance_projection_u)
        return float(result.point_target[2] - result.point_base[2])

    def compute_distance_factory(
        self,
        result_distance: Callable[[], DistanceResult],
        target: float,
    ) -> Callable[[], float]:
        """Build a residual ``selected_distance - target`` without arguments.

        Args:
            result_distance: Callable returning the current `DistanceResult`.
                It is evaluated on each call of the residual.
            target: Target value of the selected distance (meters).

        Returns:
            A callable returning ``selected_distance - target``.
        """
        return lambda: self.select_distance(result_distance()) - target
