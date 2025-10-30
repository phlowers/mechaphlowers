# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from typing import List

import numpy as np
import pint

unit = pint.UnitRegistry()

Q_ = unit.Quantity


def convert_mass_to_weight(mass: np.ndarray | List) -> np.ndarray:
    """Convert mass in kg to weight in N

    Args:
        mass (np.ndarray): mass value in kg to convert

    Returns:
        np.ndarray: weight value in N
    """
    return np.array(mass) * 9.81


def convert_weight_to_mass(weight: np.ndarray | List) -> np.ndarray:
    """Convert weight in N to mass in kg

    Args:
        mass (np.ndarray): weight value in N to convert

    Returns:
        np.ndarray: mass value in kg
    """
    return np.array(weight) / 9.81
