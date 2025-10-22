# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.models.external_loads import (
    CableLoads,
    WindSpeedPressureConverter,
)
from mechaphlowers.entities.arrays import WeatherArray
from mechaphlowers.entities.data_container import DataContainer

NB_SPAN = 3


@pytest.fixture
def cable_data_dict() -> dict:
    # units are incorrect
    return {
        "diameter": 22.4,
        "linear_weight": 9.55494,
    }


def test_compute_ice_load(cable_data_dict: dict) -> None:
    cable_data_dict.update(
        {
            "ice_thickness": np.array([1, 2.1, 0.0]),
            "wind_pressure": np.zeros(NB_SPAN),
        }
    )
    weather_loads = CableLoads(**cable_data_dict)

    weather_loads.ice_load


def test_compute_wind_load(cable_data_dict: dict) -> None:
    cable_data_dict.update(
        {
            "ice_thickness": np.array([1, 2.1, 0.0]),
            "wind_pressure": np.array([240.12, 0, -240.13]),
        }
    )
    weather_loads = CableLoads(**cable_data_dict)

    weather_loads.wind_load


def test_total_load_coefficient_and_angle(cable_data_dict: dict) -> None:
    cable_data_dict.update(
        {
            "ice_thickness": np.array([1, 2.1, 0.0]),
            "wind_pressure": np.array([240.12, 0, -240.13]),
        }
    )
    weather_loads = CableLoads(**cable_data_dict)

    weather_loads.load_coefficient
    weather_loads.load_angle


def test_total_load_coefficient__data_container(
    default_data_container_one_span: DataContainer,
) -> None:
    weather = WeatherArray(
        pd.DataFrame(
            {
                "ice_thickness": [1, 2.1],
                "wind_pressure": [240.12, 0],
            }
        )
    )
    default_data_container_one_span.add_weather_array(weather)
    weather_loads = CableLoads(**default_data_container_one_span.__dict__)

    weather_loads.load_coefficient
    weather_loads.load_angle


def test_build_converter_with_gust():
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
    wind_converter.speed_average
    wind_converter.pressure
    wind_converter.pressure_rounded


def test_build_converter_with_average_wind():
    speed_average_wind_open_country = np.array([2, 7])
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
        wind_converter.speed_average,
        speed_average_wind_open_country,
    )
    wind_converter.pressure
    wind_converter.pressure_rounded


def test_build_converter_no_wind():
    wind_angle_cable_degrees = np.array([90, 70])
    tower_height = np.array([20, 50])
    voltage = 90
    with pytest.raises(TypeError):
        WindSpeedPressureConverter(
            angle_cable_degrees=wind_angle_cable_degrees,
            tower_height=tower_height,
            voltage=voltage,
        )


def test_build_converter_both_wind_values():
    # test that if both gust_wind and speed_average_wind_open_country are provided,
    # the speed_average_wind_open_country is used for pressure calculation
    gust_wind = np.array([0, 0])
    speed_average_wind_open_country = np.array([13, 9])
    wind_angle_cable_degrees = np.array([90, 70])
    tower_height = np.array([20, 50])
    voltage = 90
    wind_converter = WindSpeedPressureConverter(
        tower_height=tower_height,
        gust=gust_wind,
        speed_average_open_country=speed_average_wind_open_country,
        angle_cable_degrees=wind_angle_cable_degrees,
        voltage=voltage,
    )
    np.testing.assert_equal(
        wind_converter.speed_average,
        speed_average_wind_open_country,
    )
    wind_converter.pressure
    np.testing.assert_equal(
        wind_converter.pressure_rounded,
        np.array([230, 100]),
    )
