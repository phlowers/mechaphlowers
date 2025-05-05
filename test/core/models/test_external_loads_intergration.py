# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.models.external_loads import (
    WindSpeedPressureConverter,
)


def test_convert_gust_wind():
    gust_wind = np.array([50, 30])
    wind_angle_cable_degrees = np.array([90, 70])
    tower_height = np.float64(50)
    voltage = np.float64(90)
    wind_converter = WindSpeedPressureConverter(
        wind_angle_cable_degrees, tower_height, voltage, gust_wind=gust_wind
    )
    np.testing.assert_equal(
        wind_converter.speed_average_wind_open_country, np.array([9, 5.4])
    )
    np.testing.assert_equal(
        wind_converter.get_pressure_rounded(), np.array([110, 40])
    )
    assert True


def test_convert_average_wind():
    speed_average_wind_open_country = np.array([7, 10])
    wind_angle_cable_degrees = np.array([90, 70])
    tower_height = np.float64(50)
    voltage = np.float64(90)
    wind_converter = WindSpeedPressureConverter(
        wind_angle_cable_degrees,
        tower_height,
        voltage,
        speed_average_wind_open_country=speed_average_wind_open_country,
    )
    np.testing.assert_equal(
        wind_converter.get_pressure_rounded(), np.array([70, 120])
    )
    assert True
