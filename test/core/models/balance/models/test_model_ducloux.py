# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd

from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.core.models.balance.models.model_ducloux import (
    nodes_builder,
)
from mechaphlowers.entities.arrays import CableArray, SectionArray


def test_section_array_to_nodes(section_array_complete: SectionArray):
    nodes_builder(section_array_complete)


def test_load_span_model(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": [0, 10, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [100, 50, 500, 0],
                "load_mass": [0, 500, 0, np.nan],
                "load_position": [0.2, 0.4, 0.6, np.nan],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})

    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array,
    )

    balance_engine.solve_adjustment()

    balance_engine.solve_change_state()
    nodes_span_model = balance_engine.balance_model.nodes_span_model
    assert nodes_span_model.parameter.shape == (5,)
    np.testing.assert_equal(
        nodes_span_model.span_index, np.array([0, 1, 1, 2, 3])
    )
    np.testing.assert_equal(
        nodes_span_model.span_type, np.array([0, 1, 2, 0, 0])
    )
