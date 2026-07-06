# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.data.units import convert_mass_to_weight
from mechaphlowers.utils import arr


class SpanLoads:
    def __init__(
        self,
        load_position_distance: np.ndarray | list,
        load_mass: np.ndarray | list,
        span_length: np.ndarray | list,
    ):
        """Create an oject to store discrete loads.

        NB: The length of load_position_distance and load_mass must be the number of spans.
        The length of span_length must be the number of pylons (same as SectionArray.data.span_length).
        """
        self.set_loads(load_position_distance, load_mass, span_length)

    def set_loads(
        self,
        load_position_distance: np.ndarray | list,
        load_mass: np.ndarray | list,
        span_length: np.ndarray | list,
    ) -> None:
        """Set loads.

        Input for position is a distance, and will be converted into ratio.

        Warning: expected lengths for the arguments are different. Expected length of load_position_distance and
        load_mass is the number of spans. Each value refers to a span. Expected length of span_length is the number
        of pylons (same as SectionArray.data.span_length). Last element should be nan or zero.

        If either load_position_distance[i] or load_mass[i] is 0 or nan, it means there is no load at span i.

        Args:
            load_position_distance (np.ndarray | list): Position of the loads, in meters
            load_mass (np.ndarray | list): Mass of the loads

        Raises:
            ValueError: if at least one load_position_distance is not in [0, span_length],
                or if the arguments don't have the right lengths.
        """
        load_position_distance, load_mass, span_length = (
            self._validate_load_arguments(
                load_position_distance,
                load_mass,
                span_length,
            )
        )
        self.load_position = self._compute_load_position_ratio(
            load_position_distance, span_length
        )
        self.load_mass = load_mass

    @staticmethod
    def _validate_load_arguments(
        load_position_distance: np.ndarray | list,
        load_mass: np.ndarray | list,
        span_length: np.ndarray | list,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(load_position_distance) != len(load_mass):
            raise ValueError(
                "load_position_distance and load_mass must have the same length. "
                f"Got {len(load_position_distance)=} and {len(load_mass)=}"
            )
        if len(load_position_distance) + 1 != len(span_length):
            raise ValueError(
                "Length of load_position_distance must be length of span_length - 1. "
                f"Got {len(load_position_distance)=} and {len(span_length)=}"
            )

        load_position_distance = np.array(load_position_distance)
        span_length = np.array(span_length)
        if (load_position_distance > arr.decr(span_length)).any() or (
            load_position_distance < 0
        ).any():
            raise ValueError(
                f"{load_position_distance=} should be all between 0 and {span_length=}"
            )
        load_mass = np.array(load_mass)
        return load_position_distance, load_mass, span_length

    @staticmethod
    def _compute_load_position_ratio(
        load_position_distance: np.ndarray, span_length: np.ndarray
    ) -> np.ndarray:
        # Length of span_length must be length of load_position_distance + 1
        # This formula for load_position_ratio may change later
        return load_position_distance / arr.decr(span_length)

    @property
    def load_weight(self) -> np.ndarray:
        return convert_mass_to_weight(self.load_mass)

    @property
    def has_load_on_span(self) -> np.ndarray:
        return self.compute_has_load_on_span(
            self.load_weight, self.load_position
        )

    @staticmethod
    def compute_has_load_on_span(load_weight, load_position) -> np.ndarray:
        return (
            np.logical_not(np.isnan(load_weight))
            & np.logical_not(np.isnan(load_position))
            & (load_weight != 0)
            & (load_position != 0)
        )
