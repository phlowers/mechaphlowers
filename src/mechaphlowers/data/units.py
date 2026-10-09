# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from typing import overload

import numpy as np
import pint
from pint import Quantity, UnitRegistry

from mechaphlowers.utils import Number

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

__all__ = ["unit", "Q_", "Quantity", "convert_angle_to_rad"]


def convert_weight_to_mass(weight: np.ndarray | list) -> np.ndarray:
    """Convert weight (N) to mass (kg)

    Args:
        weight (np.ndarray | list): weight value in N to convert

    Returns:
        np.ndarray: mass value in kg
    """
    return Q_(np.array(weight), "N").to("kg").magnitude


def convert_mass_to_weight(mass: np.ndarray | list) -> np.ndarray:
    """Convert mass (kg) to weight in (N)

    Args:
        mass (np.ndarray | list): mass value in kg to convert

    Returns:
        np.ndarray: weight value in N
    """
    return Q_(np.array(mass), "kg").to("N").magnitude


@overload
def convert_angle_to_rad(
    angle: Number,
    input_unit: str = "grad",
) -> Number: ...


@overload
def convert_angle_to_rad(
    angle: np.ndarray,
    input_unit: str = "grad",
) -> np.ndarray: ...


def convert_angle_to_rad(
    angle: np.ndarray | Number,
    input_unit: str = "grad",
) -> np.ndarray | Number:
    """Converts angle(s) to radians.

    Args:
        angle (np.ndarray | Number): angle or array of angles
        input_unit (str): unit of the angle(s). Defaults to "grad".

    Returns:
        np.ndarray | Number: angle or array of angles in radians
    """
    return Q_(angle, input_unit).to("rad").magnitude
