# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from pandera.typing import pandas as pdt

from mechaphlowers.core.models.external_loads import WeatherLoads
from mechaphlowers.entities.arrays import CableArray, WeatherArray

NB_SPAN = 3


@pytest.fixture
def cable() -> CableArray:
	return CableArray(
		pdt.DataFrame(
			{
				"section": np.full(NB_SPAN, [345.5]),
				"diameter": np.full(NB_SPAN, [22.4]),
				"linear_weight": np.full(NB_SPAN, [9.6]),
				"young_modulus": np.full(NB_SPAN, [59]),
				"dilatation_coefficient": np.full(NB_SPAN, [23]),
				"temperature_reference": np.full(NB_SPAN, [15]),
			}
		)
	)


def test_compute_ice_load(cable: CableArray) -> None:
	weather = WeatherArray(
		pdt.DataFrame(
			{
				"ice_thickness": [0.01, 0.02, 0.0],
				"wind_pressure": np.zeros(NB_SPAN),
			}
		)
	)
	weather_loads = WeatherLoads(
		cable,
		weather,
	)

	weather_loads.ice_load()


def test_compute_wind_load(cable: CableArray) -> None:
	weather = WeatherArray(
		pdt.DataFrame(
			{
				"ice_thickness": [0.01, 0.02, 0.0],
				"wind_pressure": [240.12, 0, -240.13],
			}
		)
	)
	weather_loads = WeatherLoads(
		cable,
		weather,
	)

	weather_loads.wind_load()


def test_total_load_coefficient_and_angle(cable: CableArray) -> None:
	weather = WeatherArray(
		pdt.DataFrame(
			{
				"ice_thickness": [0.01, 0.02, 0.0],
				"wind_pressure": [240.12, 0, -240.13],
			}
		)
	)
	weather_loads = WeatherLoads(
		cable,
		weather,
	)

	result = weather_loads.result()
	result.load_coefficient
	result.load_angle
