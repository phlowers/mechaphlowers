# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.entities.arrays import (
	CableArray,
	SectionArray,
	WeatherArray,
)
from mechaphlowers.entities.data_container import (
	DataContainer,
	factory_data_container,
)

projet_dir: Path = Path(__file__).resolve().parents[1]
source_dir: Path = projet_dir / "src"
sys.path.insert(0, str(source_dir))


@pytest.fixture
def factory_cable_array() -> Callable[[int], CableArray]:
	def _method_cable_array(nb_span=2):
		return CableArray(
			pd.DataFrame(
				{
					"section": [345.55] * nb_span,
					"diameter": [22.4] * nb_span,
					"linear_weight": [9.55494] * nb_span,
					"young_modulus": [59] * nb_span,
					"dilatation_coefficient": [23] * nb_span,
					"temperature_reference": [15] * nb_span,
					"a0": [0] * nb_span,
					"a1": [59] * nb_span,
					"a2": [0] * nb_span,
					"a3": [0] * nb_span,
					"a4": [0] * nb_span,
					"b0": [0] * nb_span,
					"b1": [0] * nb_span,
					"b2": [0] * nb_span,
					"b3": [0] * nb_span,
					"b4": [0] * nb_span,
				}
			)
		)

	return _method_cable_array


@pytest.fixture
def default_section_array_two_spans() -> SectionArray:
	return SectionArray(
		pd.DataFrame(
			{
				"name": ["1", "2"],
				"suspension": [False, False],
				"conductor_attachment_altitude": [30, 40],
				"crossarm_length": [0, 0],
				"line_angle": [0, 0],
				"insulator_length": [0, 0],
				"span_length": [480, np.nan],
			}
		)
	)


@pytest.fixture
def default_section_array_four_spans() -> SectionArray:
	return SectionArray(
		pd.DataFrame(
			{
				"name": np.array(["support 1", "2", "three", "support 4"]),
				"suspension": np.array([False, True, True, False]),
				"conductor_attachment_altitude": np.array([2.2, 5, -0.12, 0]),
				"crossarm_length": np.array([10, 12.1, 10, 10.1]),
				"line_angle": np.array([0, 360, 90.1, -90.2]),
				"insulator_length": np.array([0, 4, 3.2, 0]),
				"span_length": np.array([400, 500.2, 500.0, np.nan]),
			}
		)
	)


@pytest.fixture
def factory_neutral_weather_array() -> Callable[[int], WeatherArray]:
	def _method_cable_array(nb_span=2):
		return CableArray(
			pd.DataFrame(
				{
					"ice_thickness": [0.0] * nb_span,
					"wind_pressure": [0.0] * nb_span,
				}
			)
		)

	return _method_cable_array


@pytest.fixture
def default_data_container_two_spans() -> DataContainer:
	NB_SPAN = 2
	cable_array = CableArray(
		pd.DataFrame(
			{
				"section": [345.55] * NB_SPAN,
				"diameter": [22.4] * NB_SPAN,
				"linear_weight": [9.55494] * NB_SPAN,
				"young_modulus": [59] * NB_SPAN,
				"dilatation_coefficient": [23] * NB_SPAN,
				"temperature_reference": [15] * NB_SPAN,
				"a0": [0] * NB_SPAN,
				"a1": [59] * NB_SPAN,
				"a2": [0] * NB_SPAN,
				"a3": [0] * NB_SPAN,
				"a4": [0] * NB_SPAN,
				"b0": [0] * NB_SPAN,
				"b1": [0] * NB_SPAN,
				"b2": [0] * NB_SPAN,
				"b3": [0] * NB_SPAN,
				"b4": [0] * NB_SPAN,
			}
		)
	)

	section_array = SectionArray(
		pd.DataFrame(
			{
				"name": ["1", "2"],
				"suspension": [False, False],
				"conductor_attachment_altitude": [30, 40],
				"crossarm_length": [0, 0],
				"line_angle": [0, 0],
				"insulator_length": [0, 0],
				"span_length": [480, np.nan],
				"sagging_parameter": [2000, 2000],
				"sagging_temperature": [15, 15],
			}
		)
	)
	section_array.sagging_parameter = 2000
	section_array.sagging_temperature = 15

	weather_array = WeatherArray(
		pd.DataFrame(
			{
				"ice_thickness": [0.0, 0.0],
				"wind_pressure": np.zeros(NB_SPAN),
			}
		)
	)

	data_container = factory_data_container(
		section_array, cable_array, weather_array
	)
	return data_container
