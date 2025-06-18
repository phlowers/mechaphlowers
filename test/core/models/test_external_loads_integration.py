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
    tower_height = np.array([20, 50])
    voltage = 90
    wind_converter = WindSpeedPressureConverter(
        tower_height,
        gust=gust_wind,
        angle_cable_degrees=wind_angle_cable_degrees,
        voltage=voltage,
    )
    np.testing.assert_equal(wind_converter.speed_average, np.array([9, 5.4]))
    np.testing.assert_equal(
        wind_converter.pressure_rounded, np.array([110, 40])
    )


def test_convert_average_wind():
    speed_average_wind_open_country = np.array([7, 10])
    wind_angle_cable_degrees = np.array([90, 70])
    tower_height = np.array([20, 50])
    voltage = 90
    wind_converter = WindSpeedPressureConverter(
        tower_height,
        speed_average_open_country=speed_average_wind_open_country,
        angle_cable_degrees=wind_angle_cable_degrees,
        voltage=voltage,
    )
    np.testing.assert_equal(
        wind_converter.pressure_rounded, np.array([70, 120])
    )


def test_convert_work_high_voltage():
    speed_average_wind_open_country = np.array([7, 10])
    wind_angle_cable_degrees = np.array([60, 70])
    tower_height = np.array([40, 70])
    voltage = 400
    category_surface_roughness = "IIIa"
    wind_converter = WindSpeedPressureConverter(
        tower_height,
        speed_average_open_country=speed_average_wind_open_country,
        angle_cable_degrees=wind_angle_cable_degrees,
        voltage=voltage,
        category_surface_roughness=category_surface_roughness,
        work=True,
    )
    np.testing.assert_equal(
        wind_converter.pressure_rounded, np.array([50, 130])
    )
