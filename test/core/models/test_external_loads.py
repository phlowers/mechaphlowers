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
        "linear_weight": 9.6,
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
    tower_height = np.float64(50)
    voltage = np.float64(90)
    wind_converter = WindSpeedPressureConverter(
        wind_angle_cable_degrees, tower_height, voltage, gust_wind=gust_wind
    )
    wind_converter.speed_average_wind_open_country
    wind_converter.get_pressure()
    wind_converter.get_pressure_rounded()


def test_build_converter_with_average_wind():
    speed_average_wind_open_country = np.array([2, 7])
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
        wind_converter.speed_average_wind_open_country,
        speed_average_wind_open_country,
    )
    wind_converter.get_pressure()
    wind_converter.get_pressure_rounded()


def test_build_converter_no_wind():
    wind_angle_cable_degrees = np.array([90, 70])
    tower_height = np.float64(50)
    voltage = np.float64(90)
    with pytest.raises(TypeError):
        WindSpeedPressureConverter(
            wind_angle_cable_degrees, tower_height, voltage
        )
