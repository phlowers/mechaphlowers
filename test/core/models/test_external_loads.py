# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import WeatherArray
from mechaphlowers.entities.data_container import DataContainer

NB_SPAN = 3


@pytest.fixture
def cable_data_dict() -> dict:
	# units are incorrect
	return {
		"section": np.full(NB_SPAN, [345.5]),
		"diameter": np.full(NB_SPAN, [22.4]),
		"linear_weight": np.full(NB_SPAN, [9.6]),
		"young_modulus": np.full(NB_SPAN, [59]),
		"dilatation_coefficient": np.full(NB_SPAN, [23]),
		"temperature_reference": np.full(NB_SPAN, [15]),
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
	default_data_container_two_spans: DataContainer,
) -> None:
	weather = WeatherArray(
		pd.DataFrame(
			{
				"ice_thickness": [1, 2.1],
				"wind_pressure": [240.12, 0],
			}
		)
	)
	default_data_container_two_spans.add_weather_array(weather)
	weather_loads = CableLoads(**default_data_container_two_spans.__dict__)

	weather_loads.load_coefficient
	weather_loads.load_angle
