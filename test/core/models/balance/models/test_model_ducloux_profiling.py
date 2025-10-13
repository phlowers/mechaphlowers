# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import random

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.models.balance.engine import (
    BalanceEngine,
)
from mechaphlowers.data.units import Q_
from mechaphlowers.entities.arrays import CableArray, SectionArray


@pytest.mark.skip(reason="This is a performance test")
def test_load_all_spans_wind_ice_temp_profiling():
    cable_AM600 = CableArray(
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

    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": Q_(np.array([0, 10, 0, 0]), "grad")
                .to("deg")
                .magnitude,
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_weight": [1000, 500, 500, 1000],
                "load_weight": [500, 1000, 500, np.nan],
                "load_position": [0.2, 0.4, 0.6, np.nan],
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    section_3d_angles_arm = BalanceEngine(
        cable_array=cable_AM600, section_array=section_array
    )

    section_3d_angles_arm.solve_adjustment()

    for i in range(10):
        new_temperature = np.array([random.randrange(-40, 90)] * 3)
        ice_thickness = np.array([random.randrange(0, 5)] * 4) * 1e-2
        wind_pressure = np.array([random.randrange(0, 700)] * 4)
        section_3d_angles_arm.solve_change_state(
            wind_pressure, ice_thickness, new_temperature
        )
        print(i)
    print("finished")


# code to launch from command line

if __name__ == "__main__":
    test_load_all_spans_wind_ice_temp_profiling()
