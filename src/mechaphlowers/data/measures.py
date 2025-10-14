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

from mechaphlowers.core.papoto.papoto_model import (
    papoto_2_points,
    papoto_validity,
)


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
    def validity(self, *args, **kwargs):
        """Method to compute validity criteria."""
        raise NotImplementedError("Validity criteria method not implemented.")

    def check_validity(self, *args, **kwargs):
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
        a: np.ndarray,
        HL: np.ndarray,
        VL: np.ndarray,
        HR: np.ndarray,
        VR: np.ndarray,
        H1: np.ndarray,
        V1: np.ndarray,
        H2: np.ndarray,
        V2: np.ndarray,
        H3: np.ndarray,
        V3: np.ndarray,
    ):
        """Compute the PAPOTO measure.

        Args:
            a (np.ndarray): Length of the span
            HL (np.ndarray): horizontal distance of the left part of the span
            VL (np.ndarray): vertical distance of the left part of the span
            HR (np.ndarray): horizontal distance of the right part of the span
            VR (np.ndarray): vertical distance of the right part of the span
            H1 (np.ndarray): horizontal distance of point 1
            V1 (np.ndarray): vertical distance of point 1
            H2 (np.ndarray): horizontal distance of point 2
            V2 (np.ndarray): vertical distance of point 2
            H3 (np.ndarray): horizontal distance of point 3
            V3 (np.ndarray): vertical distance of point 3
        Returns:
            None
        """

        self.measures = {
            "a": a,
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

        measures_converted = self.input_conversion(self.measures)

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

    def _conversion_function(self, value):
        """Convert inputs to the required format."""
        # TODO: implement final version with pint
        return value

    def input_conversion(
        self, data: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Convert inputs to the required format."""
        for key, value in data.items():
            data[key] = self._conversion_function(value)
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


papoto = PapotoParameterMeasure()
