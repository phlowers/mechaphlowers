# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np

from mechaphlowers.core.models.cable.thermal import get_cable_temperature
from mechaphlowers.entities.arrays import CableArray


def test_thermohl_cable_temp_floats(cable_array_AM600: CableArray):
    get_cable_temperature(
        cable_array_AM600,
        latitude=45.0,
        longitude=0.0,
        altitude=0.0,
        azimuth=0.0,
        month=3,
        day=21,
        hour=12,
        intensity=100.0,
        ambient_temp=15.0,
        wind_speed=0.0,
        wind_angle=90.0,
    )


def test_thermohl_cable_temp_arrays(cable_array_AM600: CableArray):
    get_cable_temperature(
        cable_array_AM600,
        latitude=np.array([45.0, 45.0]),
        longitude=np.array([0.0, 0.0]),
        altitude=np.array([0.0, 0.0]),
        azimuth=np.array([0.0, 0.0]),
        month=np.array(
            [
                3,
                3,
            ]
        ),
        day=np.array(
            [
                21,
                21,
            ]
        ),
        hour=np.array(
            [
                12,
                12,
            ]
        ),
        intensity=np.array([100.0, 100.0]),
        ambient_temp=np.array([15.0, 15.0]),
        wind_speed=np.array([0.0, 0.0]),
        wind_angle=np.array(
            [
                90.0,
                90.0,
            ]
        ),
    )
