# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from typing import List

import numpy as np
import pint
from pint import Quantity, UnitRegistry

unit = UnitRegistry()

c = pint.Context("mecha")
c.add_transformation(
    "kg",
    "N",
    lambda unit, x: x * unit.Quantity(9.81, "m/s^2"),  # type: ignore
)
c.add_transformation(
    "N",
    "kg",
    lambda unit, x: x / unit.Quantity(9.81, "m/s^2"),  # type: ignore
)
unit.add_context(c)
unit.enable_contexts("mecha")

Q_ = unit.Quantity

__all__ = ["unit", "Q_", "Quantity"]


def convert_weight_to_mass(weight: np.ndarray | List) -> np.ndarray:
    """Convert weight in N to mass in kg

    Args:
        mass (np.ndarray): weight value in N to convert

    Returns:
        np.ndarray: mass value in kg
    """
    return Q_(np.array(weight), "N").to("kg").magnitude
