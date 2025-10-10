# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0






import numpy as np
import pandas as pd
from pytest import fixture
from mechaphlowers.core.models.balance.models.model_ducloux import nodes_builder
from mechaphlowers.entities.arrays import SectionArray
import mechaphlowers.data.units as f


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



def test_section_array_to_nodes(section_array_angles):
    nodes_builder(section_array_angles)
