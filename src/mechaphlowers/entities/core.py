from typing import Tuple

import numpy as np

from mechaphlowers.config import options
from mechaphlowers.data.units import Q_, Quantity


class QuantityArray:
    def __init__(
        self, value: np.ndarray, input_unit: str, output_unit: str
    ) -> None:
        self.quantity = Q_(value, input_unit).to(output_unit)
        self.input_unit = input_unit  # for debug
        self.output_unit = output_unit

    @property
    def value(self) -> np.ndarray:
        return self.quantity.m

    @property
    def unit(self) -> str:
        return str(self.quantity.u)

    @property
    def symbol(self) -> str:
        return f"{self.quantity.u:P~}"

    def to_tuple(self) -> Tuple[Quantity, str]:
        return (self.quantity.m, self.symbol)

    def __str__(self) -> str:
        return f"{self.quantity.m} {self.symbol}"

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}({self.quantity.m}, {self.symbol})"


class VhlStrength:
    output_unit = options.output_units.force

    def __init__(self, vhl: np.ndarray, input_unit="N") -> None:
        """expected format: [[V0, V1, ...], [H0, H1, ...], [L0, L1, ...]]"""
        self._vhl_section = vhl
        self.input_unit = input_unit

    @property
    def vhl_matrix(self) -> QuantityArray:
        return QuantityArray(
            self._vhl_section, self.input_unit, self.output_unit
        )

    @property
    def vhl(self) -> Tuple[QuantityArray, QuantityArray, QuantityArray]:
        return (self.V, self.H, self.L)

    @property
    def V(self) -> QuantityArray:
        return QuantityArray(
            self._vhl_section[0, :], self.input_unit, self.output_unit
        )

    @property
    def H(self) -> QuantityArray:
        return QuantityArray(
            self._vhl_section[1, :], self.input_unit, self.output_unit
        )

    @property
    def L(self) -> QuantityArray:
        return QuantityArray(
            self._vhl_section[2, :], self.input_unit, self.output_unit
        )

    def __str__(self) -> str:
        return f"V: {str(self.V)}\nH: {str(self.H)}\nL: {str(self.L)}\n"

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}\n{self.__str__()}"
