# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from typing import Tuple

import numpy as np
from pint import Quantity, UnitRegistry

unit = UnitRegistry()

Q_ = unit.Quantity


class QuantityArray:
    def __init__(
        self, array: np.ndarray, input_unit: str, output_unit: str
    ) -> None:
        self.quantity = Q_(array, input_unit).to(output_unit)
        self.input_unit = input_unit  # for debug
        self.output_unit = output_unit

    @property
    def array(self) -> np.ndarray:
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
