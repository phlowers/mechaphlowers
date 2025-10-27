# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd
import pytest
from pytest import fixture

from mechaphlowers.core.models.balance.engine import (
    BalanceEngine,
)
from mechaphlowers.data.units import Q_
from mechaphlowers.entities.arrays import CableArray, SectionArray


@fixture
def cable_array_AM600() -> CableArray:
    return CableArray(
        pd.DataFrame(
            {
                "section": [600.4],
                "diameter": [31.86],
                "linear_weight": [17.658],
                "young_modulus": [60],
                "dilatation_coefficient": [23],
                "temperature_reference": [15],
                "a0": [0],
                "a1": [60],
                "a2": [0],
                "a3": [0],
                "a4": [0],
                "b0": [0],
                "b1": [0],
                "b2": [0],
                "b3": [0],
                "b4": [0],
            }
        )
    )


@fixture
def balance_engine_simple(cable_array_AM600: CableArray) -> BalanceEngine:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 0, 0, 0],
                "line_angle": Q_(np.array([0, 0, 0, 0]), "grad")
                .to('deg')
                .magnitude,
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_weight": [1000, 500, 500, 1000],
                "load_weight": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
    return BalanceEngine(
        cable_array=cable_array_AM600, section_array=section_array
    )


@fixture
def section_array_angles() -> SectionArray:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": Q_(np.array([0, 0, 0, 0]), "grad")
                .to('rad')
                .magnitude,
                "insulator_length": [0, 3, 3, 0],
                "span_length": [500, 300, 400, np.nan],
                "insulator_weight": [1000, 500, 500, 1000],
                "load_weight": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
    return section_array


def test_element_initialisation(balance_engine_simple: BalanceEngine):
    print("\n")
    print(balance_engine_simple.balance_model)


def test_element_change_state(balance_engine_simple: BalanceEngine):
    with pytest.raises(AttributeError):
        balance_engine_simple.solve_change_state()

    balance_engine_simple.solve_adjustment()

    balance_engine_simple.solve_change_state()


def test_change_state_size(balance_engine_simple: BalanceEngine):
    balance_engine_simple.solve_adjustment()
    with pytest.raises(AttributeError):
        balance_engine_simple.solve_change_state(
            wind_pressure=np.array([1, 1, 1])
        )  # array size should be 4
    with pytest.raises(AttributeError):
        balance_engine_simple.solve_change_state(
            ice_thickness=np.array([1, 1, 1])
        )
    with pytest.raises(AttributeError):
        balance_engine_simple.solve_change_state(
            new_temperature=np.array([1, 1, 1])
        )
