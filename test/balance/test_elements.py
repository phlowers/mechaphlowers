# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd
from pytest import fixture

import mechaphlowers.core.models.balance.functions as f
from mechaphlowers.core.models.balance.elements import (
    BalanceEngine,
    Cable,
    section_array_to_nodes,
)
from mechaphlowers.entities.arrays import SectionArray


@fixture
def cable_AM600():
    return Cable(600.4e-6, 17.658, 0.000023, 60e9, 31.86e-3, 320)


@fixture
def balance_engine_simple(cable_AM600: Cable) -> BalanceEngine:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 0, 0, 0],
                "line_angle": f.grad_to_deg(np.array([0, 0, 0, 0])),
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
    return BalanceEngine(cable=cable_AM600, section_array=section_array)


@fixture
def section_array_angles() -> SectionArray:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": f.grad_to_rad(np.array([0, 10, 0, 0])),
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
    # load = section_2d_note.nodes.load

    print("\n")
    print(balance_engine_simple.balance_model)
    print(balance_engine_simple.balance_model.nodes)


def test_element_change_state(balance_engine_simple: BalanceEngine):
    balance_engine_simple.solve_adjustment()

    balance_engine_simple.solve_change_state()
    assert True


def test_section_array_to_nodes(section_array_angles):
    section_array_to_nodes(section_array_angles)
