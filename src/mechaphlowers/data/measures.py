# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


"""Measurung module

This module provides functions to compute various measures on sections and spans.
"""

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

from mechaphlowers.config import options
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.core.papoto.papoto_model import (
    papoto_2_points,
    papoto_validity,
)
from mechaphlowers.data.units import Q_
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.utils import float_to_array


class ParameterMeasure(ABC):
    """Class to compute measures on parameters."""

    @abstractmethod
    def measure_method(self, *args, **kwargs):
        """Abstract method to be implemented by subclasses."""
        pass

    def uncertainty(self, *args, **kwargs):
        """Method to compute uncertainty."""
        raise NotImplementedError("Uncertainty method not implemented.")

    @property
    def validity(self):
        """Method to compute validity criteria."""
        raise NotImplementedError("Validity criteria method not implemented.")

    def check_validity(self):
        """Method to check validity of the measure."""
        raise NotImplementedError("Check validity method not implemented.")

    @property
    @abstractmethod
    def parameter(self):
        """Property to get the computed parameter."""
        pass


class PapotoParameterMeasure(ParameterMeasure):
    """Class to compute PAPOTO parameter measures."""

    validity_criteria = 0.005

    def measure_method(
        self,
        a: np.ndarray | float | int,
        HL: np.ndarray | float | int,
        VL: np.ndarray | float | int,
        HR: np.ndarray | float | int,
        VR: np.ndarray | float | int,
        H1: np.ndarray | float | int,
        V1: np.ndarray | float | int,
        H2: np.ndarray | float | int,
        V2: np.ndarray | float | int,
        H3: np.ndarray | float | int,
        V3: np.ndarray | float | int,
        angle_unit: str = "grad",
    ):
        """Compute the PAPOTO measure.

        Args:
            a (np.ndarray): Length of the span
            HL (np.ndarray): horizontal angle of the left part of the span
            VL (np.ndarray): vertical angle of the left part of the span
            HR (np.ndarray): horizontal angle of the right part of the span
            VR (np.ndarray): vertical angle of the right part of the span
            H1 (np.ndarray): horizontal angle of point 1
            V1 (np.ndarray): vertical angle of point 1
            H2 (np.ndarray): horizontal angle of point 2
            V2 (np.ndarray): vertical angle of point 2
            H3 (np.ndarray): horizontal angle of point 3
            V3 (np.ndarray): vertical angle of point 3
        Returns:
            None
        """

        self.measures = {
            "HL": HL,
            "VL": VL,
            "HR": HR,
            "VR": VR,
            "H1": H1,
            "V1": V1,
            "H2": H2,
            "V2": V2,
            "H3": H3,
            "V3": V3,
        }
        self.measures = float_to_array(self.measures)
        self.angle_unit = angle_unit
        measures_converted = self.input_conversion(self.measures)
        measures_converted["a"] = a

        self.parameter_1_2 = papoto_2_points(
            **self.select_points_in_dict(1, 2, measures_converted)
        )
        self.parameter_2_3 = papoto_2_points(
            **self.select_points_in_dict(2, 3, measures_converted)
        )
        self.parameter_1_3 = papoto_2_points(
            **self.select_points_in_dict(1, 3, measures_converted)
        )
        self._parameter = np.mean(
            np.array(
                [self.parameter_1_2, self.parameter_2_3, self.parameter_1_3]
            ),
            axis=0,
        )
        self._validity = papoto_validity(
            self.parameter_1_2, self.parameter_2_3, self.parameter_1_3
        )

    @staticmethod
    def select_points_in_dict(point_1, point_2, data):
        """Select points from the input array."""
        output_data = {
            key: value
            for key, value in data.items()
            if key not in ('H1', 'V1', 'H2', 'V2', 'H3', 'V3')
        }
        output_data["H1"], output_data["V1"] = (
            data[f"H{point_1}"],
            data[f"V{point_1}"],
        )
        output_data["H2"], output_data["V2"] = (
            data[f"H{point_2}"],
            data[f"V{point_2}"],
        )
        return output_data

    def input_conversion(self, data: Dict) -> Dict:
        """Convert inputs to the required format."""
        for key, value in data.items():
            data[key] = Q_(value, self.angle_unit).to("rad").magnitude
        return data

    @property
    def validity(self):
        return self._validity

    def check_validity(self):
        """Check the validity of the measure."""
        return self._validity < self.validity_criteria

    @property
    def parameter(self):
        return self._parameter

    def __call__(self, *args, **kwds):
        return self.measure_method(*args, **kwds)


def param_calibration(
    measured_parameter: float,
    measured_temperature: float,
    section_array: SectionArray,
    cable_array: CableArray,
    span_index: int,
    sagging_temperature: float = 15.0,
) -> float:
    """Compute an approximation of the parameter at sagging temperature (usually 15 degrees Celsius), based on the measured parameter, at a given temperature.

    This approximation is computed using only one loop of a Newton-Raphson method, more precision is not needed.

    This method only works for one span at a time, so span index must be provided.

    Args:
        measured_parameter (float): parameter value measured, using papoto method usually
        measured_temperature (float): temperature at which the parameter was measured
        section_array (SectionArray): Section array
        cable_array (CableArray): Cable array
        span_index (int): index of the span to compute the parameter for
    """
    _ZETA = options.solver.param_calibration_zeta

    def compute_parameter(
        sagging_parameter: float,
        sagging_temperature: float,
        new_temperature: float,
    ):
        section_array.sagging_parameter = sagging_parameter
        section_array.sagging_temperature = sagging_temperature
        balance_engine = BalanceEngine(
            cable_array=cable_array, section_array=section_array
        )
        balance_engine.solve_adjustment()
        balance_engine.solve_change_state(new_temperature=new_temperature)
        return balance_engine.parameter[span_index]

    # first estimation of the parameter at sagging temperature
    param_approx = compute_parameter(
        measured_parameter, measured_temperature, sagging_temperature
    )

    # computing the delta function to be zeroed
    parameter_mes_0 = compute_parameter(
        param_approx, sagging_temperature, measured_temperature
    )
    delta = parameter_mes_0 - measured_parameter

    # computing derivative by finite difference
    parameter_mes_1 = compute_parameter(
        param_approx + _ZETA, sagging_temperature, measured_temperature
    )
    delta_1 = parameter_mes_1 - measured_parameter
    derivative = (delta_1 - delta) / _ZETA

    if derivative == 0:
        # case where we get the exact solution, avoid division by zero
        return param_approx
    else:
        # newton-raphson update
        return param_approx - delta / derivative
