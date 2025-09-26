# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mechaphlowers.core.solver.cable_state import SagTensionSolver
from mechaphlowers.entities.arrays import WeatherArray

if TYPE_CHECKING:
    from mechaphlowers.api.frames import SectionDataFrame


class StateAccessor:
    """shortcut accessor class for state calculus"""

    def __init__(self, frame: SectionDataFrame):
        self.frame: SectionDataFrame = frame
        self.sag_tension: SagTensionSolver | None = None
        self.p_after_change: np.ndarray | None = None
        self.L_after_change: np.ndarray | None = None

    def L_ref(self) -> np.ndarray:
        """L_ref values for the current temperature

        Args:
                current_temperature (float | np.ndarray): current temperature in degrees Celsius

        Raises:
                ValueError: if current_temperature is not a float or an array with the same length as the section

        Returns:
                np.ndarray: L_ref values
        """
        if self.frame.deformation is None:
            raise ValueError(
                "Deformation model is not defined: setting cable usually sets deformation model"
            )
        return self.frame.deformation.L_ref()

    def change(
        self, current_temperature: np.ndarray, weather_loads: WeatherArray
    ) -> None:
        """Change the state of the cable
        Args:
                current_temperature (np.ndarray): current temperature in degrees Celsius
        """
        if self.frame.deformation is None:
            raise ValueError(
                "Deformation model is not defined: setting cable usually sets deformation model"
            )

        sag_tension_calculation = SagTensionSolver(
            **self.frame.data_container.__dict__,
        )
        sag_tension_calculation.initial_state()
        sag_tension_calculation.change_state(
            **weather_loads.to_numpy(),
            new_temperature=current_temperature,
            solver="newton",
        )
        self.sag_tension = sag_tension_calculation
        self.p_after_change = sag_tension_calculation.p_after_change()
        self.L_after_change = sag_tension_calculation.L_after_change()
        self.T_h_after_change = sag_tension_calculation.T_h_after_change
        self.frame.span.sagging_parameter = self.p_after_change
