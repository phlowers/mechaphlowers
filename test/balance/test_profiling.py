# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import random

import numpy as np

import mechaphlowers.core.models.balance.functions as f
from mechaphlowers.core.models.balance.elements import Cable, Nodes, Span


def test_load_all_spans_wind_ice_temp_profiling():
    cable_AM600 = Cable(600.4e-6, 17.658, 0.000023, 60e9, 31.86e-3, 320)

    nodes_arm = Nodes(
        L_chain=np.array([3, 3, 3, 3]),
        weight_chain=np.array([1000, 500, 500, 1000]),
        arm_length=np.array([0, 10, -10, 0]),
        line_angle=f.grad_to_rad(np.array([0, 10, 0, 0])),
        x=np.array([0, 500, 800, 1200]),
        z=np.array([30, 50, 60, 65]),
        load=np.array([500, 1000, 500]),
        load_position=np.array([0.2, 0.4, 0.6]),
    )

    section_3d_angles_arm = Span(
        parameter=2000,
        sagging_temperature=15,
        nodes=nodes_arm,
        cable=cable_AM600,
    )

    section_3d_angles_arm.adjust()

    for i in range(10):
        section_3d_angles_arm.sagging_temperature = random.randrange(-40, 90)
        section_3d_angles_arm.cable_loads.ice_thickness = (
            np.array([random.randrange(0, 5)] * 4) * 1e-2
        )
        section_3d_angles_arm.cable_loads.wind_pressure = np.array(
            [random.randrange(0, 700)] * 4
        )
        section_3d_angles_arm.change_state()
        print(i)
    print("finished")


# code to launch from command line

if __name__ == "__main__":
    test_load_all_spans_wind_ice_temp_profiling()
