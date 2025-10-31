# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple

import numpy as np

from mechaphlowers.config import options
from mechaphlowers.data.units import Q_, Quantity


class QuantityArray:
    def __init__(
        self, value: np.ndarray, input_unit: str, output_unit: str
    ) -> None:
        """Convert a numpy array from input_unit to output_unit
        Args:
            value (np.ndarray): array of values to convert
            input_unit (str): unit of the input values
            output_unit (str): desired unit of the output values
        """
        self.quantity = Q_(value, input_unit).to(output_unit)
        self.input_unit = input_unit  # for debug
        self.output_unit = output_unit

    @property
    def value(self) -> np.ndarray:
        """Return the magnitude of the quantity array in the output unit"""
        return self.quantity.m

    @property
    def unit(self) -> str:
        """Return the unit of the quantity array as a string"""
        return str(self.quantity.u)

    @property
    def symbol(self) -> str:
        """Return the unit symbol of the quantity array as a string"""
        return f"{self.quantity.u:P~}"

    def to_tuple(self) -> Tuple[Quantity, str]:
        """Helper providing the quantity array as a tuple of (magnitude, unit symbol)"""
        return (self.quantity.m, self.symbol)

    def __str__(self) -> str:
        return f"{self.quantity.m} {self.symbol}"

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}({self.quantity.m}, {self.symbol})"


class VhlStrength:
    """Class representing the VHL forces"""

    output_unit = options.output_units.force

    def __init__(self, vhl: np.ndarray, input_unit="N") -> None:
        """
        Args:
            vhl (np.ndarray): 2D array representing VHL forces (expected format: [[V0, V1, ...], [H0, H1, ...], [L0, L1, ...]]),
            input_unit (str, optional): unit of the input vhl array. Defaults to "N".
        """
        self._vhl_section = vhl
        self.input_unit = input_unit

    @property
    def vhl_matrix(self) -> QuantityArray:
        """Return the full VHL matrix as 1 QuantityArray"""
        return QuantityArray(
            self._vhl_section, self.input_unit, self.output_unit
        )

    @property
    def vhl(self) -> Tuple[QuantityArray, QuantityArray, QuantityArray]:
        """Return the V, H, L components as a tuple of 3 QuantityArrays"""
        return (self.V, self.H, self.L)

    @property
    def V(self) -> QuantityArray:
        """Return the V component as a QuantityArray"""
        return QuantityArray(
            self._vhl_section[0, :], self.input_unit, self.output_unit
        )

    @property
    def H(self) -> QuantityArray:
        """Return the H component as a QuantityArray"""
        return QuantityArray(
            self._vhl_section[1, :], self.input_unit, self.output_unit
        )

    @property
    def L(self) -> QuantityArray:
        """Return the L component as a QuantityArray"""
        return QuantityArray(
            self._vhl_section[2, :], self.input_unit, self.output_unit
        )

    @property
    def R(self) -> QuantityArray:
        """Return the resultant force of the VHL component as a QuantityArray"""
        return QuantityArray(
            np.linalg.norm(self._vhl_section, axis=0),
            self.input_unit,
            self.output_unit,
        )

    def __str__(self) -> str:
        return f"V: {str(self.V)}\nH: {str(self.H)}\nL: {str(self.L)}\n"

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}\n{self.__str__()}"
