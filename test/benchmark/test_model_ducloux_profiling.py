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
from mechaphlowers.entities.arrays import CableArray, SectionArray


@pytest.mark.skip(reason="This is a performance test")
def test_load_all_spans_wind_ice_temp_profiling():
    cable_AM600 = CableArray(
        pd.DataFrame(
            {
                "section": [600.4],
                "diameter": [31.86],
                "linear_mass": [1.8],
                "young_modulus": [60000],
                "dilatation_coefficient": [23e-6],
                "temperature_reference": [15],
                "a0": [0],
                "a1": [60000],
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
                "line_angle": [0, 10, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [1000, 500, 500, 1000],
                "load_mass": [500, 1000, 500, np.nan],
                "load_position": [0.2, 0.4, 0.6, np.nan],
            }
        )
    )
    section_array.add_units({"line_angle": "grad"})
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    section_3d_angles_arm = BalanceEngine(
        cable_array=cable_AM600, section_array=section_array
    )

    section_3d_angles_arm.solve_adjustment()

    for i in range(10):
        new_temperature = np.array([random.randrange(-40, 90)] * 4)
        ice_thickness = np.array([random.randrange(0, 5)] * 4) * 1e-2
        wind_pressure = np.array([random.randrange(0, 700)] * 4)
        section_3d_angles_arm.solve_change_state(
            wind_pressure, ice_thickness, new_temperature
        )
        print(i)
    print("finished")


@pytest.mark.skip(reason="This is a performance test")
@pytest.mark.benchmark
def test_many_spans(cable_array_AM600: CableArray):
    nb_spans = 50
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["name"] * nb_spans,
                "suspension": [False] + [True] * (nb_spans - 2) + [False],
                "conductor_attachment_altitude": [50] * nb_spans,
                "crossarm_length": [0] * nb_spans,
                "line_angle": [0] * nb_spans,
                "insulator_length": [3] * nb_spans,
                "span_length": [500] * (nb_spans - 1) + [np.nan],
                "insulator_mass": [500 / 9.81] * nb_spans,
                "load_mass": [0] * nb_spans,
                "load_position": [0] * nb_spans,
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array,
    )

    balance_engine.solve_adjustment()

    balance_engine.solve_change_state(
        wind_pressure=np.array([-200] * nb_spans)
    )


@pytest.mark.skip(reason="This is a performance test")
@pytest.mark.benchmark
def test_many_spans_with_load(cable_array_AM600: CableArray):
    nb_spans = 10
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["name"] * nb_spans,
                "suspension": [False] + [True] * (nb_spans - 2) + [False],
                "conductor_attachment_altitude": [50] * nb_spans,
                "crossarm_length": [0] * nb_spans,
                "line_angle": [0] * nb_spans,
                "insulator_length": [3] * nb_spans,
                "span_length": [500] * (nb_spans - 1) + [np.nan],
                "insulator_mass": [500 / 9.81] * nb_spans,
                "load_mass": [500 / 9.81] * nb_spans,
                "load_position": [0.5] * nb_spans,
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array,
    )

    balance_engine.solve_adjustment()

    balance_engine.solve_change_state(
        wind_pressure=np.array([-200] * nb_spans)
    )


# code to launch from command line

if __name__ == "__main__":
    test_load_all_spans_wind_ice_temp_profiling()
