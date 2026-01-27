# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.core.models.guying import GuyingLoads, GuyingLoadsResults
from mechaphlowers.data.units import Q_
from mechaphlowers.entities.arrays import CableArray, SectionArray
from test.entities.test_arrays import section_array


section_array_flat = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 30, 30, 30],
                "crossarm_length": [0, 0, 0, 0],
                "line_angle": [0, 0, 0, 0],
                "insulator_length": [0.01, 3, 3, 0.01],
                "span_length": [400, 400, 400, np.nan],
                "insulator_mass": [0, 100, 100, 0],
                "load_mass": [0, 0, 0, np.nan],
                "load_position": [0.2, 0.4, 0.6, np.nan],
            }
        ), sagging_parameter=2000, sagging_temperature=15
    )
section_array_flat.add_units({"line_angle": "grad"})

section_array_span_change = SectionArray(
    pd.DataFrame(
        {
            "name": ["1", "2", "3", "4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [30, 30, 30, 30],
            "crossarm_length": [0, 10, 10, 0],
            "line_angle": [0, 0, 0, 0],
            "insulator_length": [0.01, 3, 3, 0.01],
            "span_length": [500, 300, 400, np.nan],
            "insulator_mass": [0, 100, 100, 0],
            "load_mass": [0, 0, 0, np.nan],
            "load_position": [0.2, 0.4, 0.6, np.nan],
        }
    )
)
section_array_span_change.add_units({"line_angle": "grad"})

section_array_complete = SectionArray(
    pd.DataFrame(
        {
            "name": ["1", "2", "3", "4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [30, 50, 60, 65],
            "crossarm_length": [0, 10, 10, 0],
            "line_angle": [0, 10, 0, 0],
            "insulator_length": [0.01, 3, 3, 0.01],
            "span_length": [500, 300, 400, np.nan],
            "insulator_mass": [100, 50, 500, 0],
            "load_mass": [0, 0, 0, np.nan],
            "load_position": [0.2, 0.4, 0.6, np.nan],
        }
    )
)
section_array_complete.add_units({"line_angle": "grad"})


expected_guying_loads_left_flat = {
    "guying_load": Q_(4119.0, "daN") ,
    "vertical_load": Q_(2676.0, "daN"),
    "longitudinal_load": Q_(0.0, "daN"),
    "guying_angle_degrees": Q_(31.0, "degrees"),
}

expected_guying_pulley_loads_left_flat = {
    "guying_load": Q_(3549.0, "daN"),
    "vertical_load": Q_(2383.0, "daN"),
    "longitudinal_load": Q_(488.0, "daN"),
    "guying_angle_degrees": Q_(31.0, "degrees"),
}

expected_guying_loads_left_span_change = {
    "guying_load": Q_(4119.0, "daN") ,
    "vertical_load": Q_(2588.0, "daN"),
    "longitudinal_load": Q_(0.0, "daN"),
    "guying_angle_degrees": Q_(31.0, "degrees"),
}

expected_guying_pulley_loads_left_span_change = {
    "guying_load": Q_(3535.0, "daN"),
    "vertical_load": Q_(2836.0, "daN"),
    "longitudinal_load": Q_(1043.0, "daN"),
    "guying_angle_degrees": Q_(31.0, "degrees"),
}

expected_guying_loads_left_complete = {
    "guying_load": Q_(5016.0, "daN"),
    "vertical_load": Q_(3888.0, "daN"),
    "longitudinal_load": Q_(0.0, "daN"),
    "guying_angle_degrees": Q_(45.2, "degrees"),
}

expected_guying_pulley_loads_left_complete = {
    "guying_load": Q_(3535.0, "daN"),
    "vertical_load": Q_(2836.0, "daN"),
    "longitudinal_load": Q_(1043.0, "daN"),
    "guying_angle_degrees": Q_(45.2, "degrees"),
}

section_array_inputs = [
    (    
    section_array_flat,expected_guying_loads_left_flat, expected_guying_pulley_loads_left_flat),
    (section_array_complete, expected_guying_loads_left_complete, expected_guying_pulley_loads_left_complete),
    (section_array_span_change, expected_guying_loads_left_span_change, expected_guying_pulley_loads_left_span_change),
]

@pytest.mark.parametrize(
    "section_array, expected_guying_loads_left, expected_guying_pulley_loads_left",
    section_array_inputs,
    ids=["flat_section_array", "span_change_section_array", "complete_section_array"],
)
def test_guying_sandbox(section_array: SectionArray, expected_guying_loads_left: dict, expected_guying_pulley_loads_left: dict ,cable_array_AM600: CableArray):

    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array,
    )
    balance_engine.solve_adjustment()
    balance_engine.solve_change_state(
        new_temperature=15,
        wind_pressure=0,
    )
    guying = GuyingLoads(balance_engine)
    
    
    guying_results = guying.get_guying_loads(
        support_index=1,
        side='left',
        with_pulley=False,
        guying_height=0,
        guying_horizontal_distance=50,
    )
    
    assert guying_results == GuyingLoadsResults(**expected_guying_loads_left)
    
    
    guying_pulley_results = guying.get_guying_loads(
        support_index=1,
        side='left',
        with_pulley=True,
        guying_height=0,
        guying_horizontal_distance=50,
    )
    
    # for v1, v2 in zip(guying_pulley_results().values, GuyingLoadsResults(**expected_guying_pulley_loads_left)().values):
    #     print(v1, v2)
    assert guying_pulley_results == GuyingLoadsResults(**expected_guying_pulley_loads_left)



    # section_array = SectionArray(
    #     pd.DataFrame(
    #         {
    #             "name": ["1", "2", "3", "4"],
    #             "suspension": [False, True, True, False],
    #             "conductor_attachment_altitude": [30, 50, 60, 65],
    #             "crossarm_length": [0, 10, 10, 0],
    #             "line_angle": [0, 10, 0, 0],
    #             "insulator_length": [3, 3, 3, 3],
    #             "span_length": [500, 300, 400, np.nan],
    #             "insulator_mass": [100, 50, 500, 0],
    #             "load_mass": [0, 500, 0, np.nan],
    #             "load_position": [0.2, 0.4, 0.6, np.nan],
    #         }
    #     )
    # )
    # section_array.add_units({"line_angle": "grad"})

    # section_array = SectionArray(
    #     pd.DataFrame(
    #         {
    #             "name": ["1", "2", "3", "4"],
    #             "suspension": [False, True, True, False],
    #             "conductor_attachment_altitude": [30, 30, 30, 30],
    #             "crossarm_length": [0, 0, 0, 0],
    #             "line_angle": [0, 0, 0, 0],
    #             "insulator_length": [0.01, 3, 3, 0.01],
    #             "span_length": [400, 400, 400, np.nan],
    #             "insulator_mass": [0, 100, 100, 0],
    #             "load_mass": [0, 0, 0, np.nan],
    #             "load_position": [0.2, 0.4, 0.6, np.nan],
    #         }
    #     )
    # )
    # section_array.add_units({"line_angle": "grad"})
    
    # expected_guying_loads_left = {
    #     "guying_load": 353.0,
    #     "vertical_load": 353.73123844,
    #     "longitudinal_load": -3531.0,
    #     "guying_angle_degrees": 45.0,
    #     "delta_altitude": -20,
    # }
    
    # expected_guying_loads_right = {
    #     "guying_load": 353.0,
    #     "vertical_load": 353.73123844,
    #     "longitudinal_load": -3531.0,
    #     "guying_angle_degrees": 45.0,
    #     "delta_altitude": -20,
    # }
    
    
    # expected_guying_pulley_loads_right = {
    #     "guying_load": 353.0,
    #     "vertical_load": 353.73123844,
    #     "longitudinal_load": -3531.0,
    #     "guying_angle_degrees": 45.0,
    #     "delta_altitude": -20,
    # }
    

    # section_array.sagging_parameter = 2000
    # section_array.sagging_temperature = 15

    # V: [353.73123844 707.49778662 707.49778662 353.73123844] daN
    # H: [0. 0. 0. 0.] daN
    # L: [ 3.53160001e+03 -9.60361649e-07  9.60005127e-07 -3.53160001e+03] daN

    # section_array = SectionArray(
    #     pd.DataFrame(
    #         {
    #             "name": ["1", "2", "3", "4"],
    #             "suspension": [False, True, True, False],
    #             "conductor_attachment_altitude": [30, 30, 30, 30],
    #             "crossarm_length": [0, 0, 0, 0],
    #             "line_angle": [0, 0, 0, 0],
    #             "insulator_length": [.01, 3, 3, .01],
    #             "span_length": [500, 300, 400, np.nan],
    #             "insulator_mass": [0, 100, 100, 0],
    #             "load_mass": [0, 0, 0, np.nan],
    #             "load_position": [0.2, 0.4, 0.6, np.nan],
    #         }
    #     )
    # )
    # section_array.add_units({"line_angle": "grad"})

    # section_array.sagging_parameter = 2000
    # section_array.sagging_temperature = 15

    #     VhlStrength
    # V: [442.58285184 707.71889124 618.86727772 353.73123844] daN
    # H: [0. 0. 0. 0.] daN
    # L: [ 3.53160001e+03 -4.77512367e-07  8.07704782e-07 -3.53160001e+03] daN

    # section_array = SectionArray(
    #     pd.DataFrame(
    #         {
    #             "name": ["1", "2", "3", "4"],
    #             "suspension": [False, True, True, False],
    #             "conductor_attachment_altitude": [30, 50, 60, 65],
    #             "crossarm_length": [0, 0, 0, 0],
    #             "line_angle": [0, 0, 0, 0],
    #             "insulator_length": [.01, 3, 3, .01],
    #             "span_length": [500, 300, 400, np.nan],
    #             "insulator_mass": [0, 100, 100, 0],
    #             "load_mass": [0, 0, 0, np.nan],
    #             "load_position": [0.2, 0.4, 0.6, np.nan],
    #         }
    #     )
    # )
    # section_array.add_units({"line_angle": "grad"})

    # section_array.sagging_parameter = 2000
    # section_array.sagging_temperature = 15

    #     VhlStrength
    # V: [300.93596818 732.27630767 692.69038294 398.05083439] daN
    # H: [0. 0. 0. 0.] daN
    # L: [ 3.53160001e+03  3.68643668e-08  8.77393177e-07 -3.53160001e+03] daN

    # section_array = SectionArray(
    #     pd.DataFrame(
    #         {
    #             "name": ["1", "2", "3", "4"],
    #             "suspension": [False, True, True, False],
    #             "conductor_attachment_altitude": [30, 50, 60, 65],
    #             "crossarm_length": [0, 0, 0, 0],
    #             "line_angle": [0, 10, 0, 0],
    #             "insulator_length": [.01, 3, 3, .01],
    #             "span_length": [500, 300, 400, np.nan],
    #             "insulator_mass": [0, 100, 100, 0],
    #             "load_mass": [0, 0, 0, np.nan],
    #             "load_position": [0.2, 0.4, 0.6, np.nan],
    #         }
    #     )
    # )
    # section_array.add_units({"line_angle": "grad"})

    # section_array.sagging_parameter = 2000
    # section_array.sagging_temperature = 15
    #     VhlStrength
    # V: [297.25978447 741.47017733 686.73861264 398.04276732] daN
    # H: [ 11.67478953 523.98880223  17.95065449   0.64599774] daN
    # L: [ 3.53158069e+03  5.13507368e-01  4.89055389e-02 -3.53159993e+03] daN

    # section_array = SectionArray(
    #     pd.DataFrame(
    #         {
    #             "name": ["1", "2", "3", "4"],
    #             "suspension": [False, True, True, False],
    #             "conductor_attachment_altitude": [30, 50, 60, 65],
    #             "crossarm_length": [0, 0, 0, 0],
    #             "line_angle": [0, 10, 0, 0],
    #             "insulator_length": [.01, 3, 3, .01],
    #             "span_length": [500, 300, 400, np.nan],
    #             "insulator_mass": [0, 100, 100, 0],
    #             "load_mass": [0, 500, 0, np.nan],
    #             "load_position": [0.2, 0.4, 0.6, np.nan],
    #         }
    #     )
    # )
    # section_array.add_units({"line_angle": "grad"})

    # section_array.sagging_parameter = 2000
    # section_array.sagging_temperature = 15

    #     VhlStrength
    # V: [ 270.59508272 1033.5245396   902.81323958  407.52304188] daN
    # H: [ 12.94774701 641.19219594  20.00663983   0.67411651] daN
    # L: [ 4200.81242104   197.97405918   -96.37927151 -4302.41078276] daN

    # section_array = SectionArray(
    #     pd.DataFrame(
    #         {
    #             "name": ["1", "2", "3", "4"],
    #             "suspension": [False, True, True, False],
    #             "conductor_attachment_altitude": [30, 50, 60, 65],
    #             "crossarm_length": [0, 0, 0, 0],
    #             "line_angle": [0, 10, 0, 0],
    #             "insulator_length": [.01, 3, 3, .01],
    #             "span_length": [500, 300, 400, np.nan],
    #             "insulator_mass": [0, 100, 100, 0],
    #             "load_mass": [0, 500, 0, np.nan],
    #             "load_position": [0.2, 0.4, 0.6, np.nan],
    #         }
    #     )
    # )
    # section_array.add_units({"line_angle": "grad"})