# # Copyright (c) 2025, RTE (http://www.rte-france.com)
# # This Source Code Form is subject to the terms of the Mozilla Public
# # License, v. 2.0. If a copy of the MPL was not distributed with this
# # file, You can obtain one at http://mozilla.org/MPL/2.0/.
# # SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
from pandera.typing import pandas as pdt

from mechaphlowers.api.frames import SectionDataFrame
from mechaphlowers.core.solver.cable_state import (
	SagTensionSolver,
)
from mechaphlowers.entities.arrays import (
	CableArray,
	SectionArray,
	WeatherArray,
)
from mechaphlowers.entities.schemas import CableArrayInput


def test_functions_to_solve() -> None:
	NB_SPAN = 4
	input_cable: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		{
			"section": [345.55] * NB_SPAN,
			"diameter": [22.4] * NB_SPAN,
			"linear_weight": [9.55494] * NB_SPAN,
			"young_modulus": [59] * NB_SPAN,
			"dilatation_coefficient": [23] * NB_SPAN,
			"temperature_reference": [0] * NB_SPAN,
		}
	)
	data_section = {
		"name": ["support 1", "2", "three", "support 4"],
		"suspension": [False, True, True, False],
		"conductor_attachment_altitude": [2.2, 5, -0.12, 0],
		"crossarm_length": [10, 12.1, 10, 10.1],
		"line_angle": [0, 360, 90.1, -90.2],
		"insulator_length": [0, 4, 3.2, 0],
		"span_length": [400, 500.2, 500.0, np.nan],
	}

	section_array = SectionArray(data=pd.DataFrame(data_section))
	section_array.sagging_parameter = 2000
	section_array.sagging_temperature = 15

	cable_array = CableArray(input_cable)

	section = SectionDataFrame(section_array)
	section.add_cable(cable_array)

	weather_array = WeatherArray(
		pdt.DataFrame(
			{
				"ice_thickness": [1, 2.1, 0.0, 0.0],
				"wind_pressure": np.zeros(NB_SPAN),
			}
		)
	)

	section.add_weather(weather_array)
	unstressed_length = section.physics.L_ref(np.array([15] * NB_SPAN))

	sag_tension_calculation = SagTensionSolver(
		section_array,
		cable_array,
		weather_array,
		unstressed_length,
	)

	weather_array_final = WeatherArray(
		pdt.DataFrame(
			{
				"ice_thickness": [1, 2.1, 0.0, 0.0],
				"wind_pressure": 0 * np.ones(NB_SPAN),
			}
		)
	)

	state_0 = sag_tension_calculation.change_state(weather_array_final)
	# checking that = section.span.T_h()
	weather_array_final = WeatherArray(
		pdt.DataFrame(
			{
				"ice_thickness": [1, 2.1, 0.0, 0.0],
				"wind_pressure": 1000 * np.ones(NB_SPAN),
			}
		)
	)

	state_1 = sag_tension_calculation.change_state(weather_array_final)
	assert True


def test_functions_to_solve_0() -> None:
	NB_SPAN = 2
	input_cable: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		{
			"section": [345.55] * NB_SPAN,
			"diameter": [22.4] * NB_SPAN,
			"linear_weight": [9.55494] * NB_SPAN,
			"young_modulus": [59] * NB_SPAN,
			"dilatation_coefficient": [23] * NB_SPAN,
			"temperature_reference": [0] * NB_SPAN,
		}
	)
	data_section = {
		"name": ["1", "2"],
		"suspension": [False, False],
		"conductor_attachment_altitude": [30, 40],
		"crossarm_length": [0, 0],
		"line_angle": [0, 0],
		"insulator_length": [0, 0],
		"span_length": [480, np.nan],
	}

	section_array = SectionArray(data=pd.DataFrame(data_section))
	section_array.sagging_parameter = 2000
	section_array.sagging_temperature = 15

	cable_array = CableArray(input_cable)

	section = SectionDataFrame(section_array)
	section.add_cable(cable_array)

	initial_weather_array = WeatherArray(
		pdt.DataFrame(
			{
				"ice_thickness": [0.0, 0.0],
				"wind_pressure": np.zeros(NB_SPAN),
			}
		)
	)

	section.add_weather(initial_weather_array)
	unstressed_length = section.physics.L_ref(np.array([15] * NB_SPAN))

	sag_tension_calculation = SagTensionSolver(
		section_array,
		cable_array,
		initial_weather_array,
		unstressed_length,
	)

	weather_array_final = WeatherArray(
		pdt.DataFrame(
			{
				"ice_thickness": [1, 1],
				"wind_pressure": 0 * np.ones(NB_SPAN),
			}
		)
	)

	state_0 = sag_tension_calculation.change_state(weather_array_final)
	# checking that = section.span.T_h()
	weather_array_final = WeatherArray(
		pdt.DataFrame(
			{
				"ice_thickness": [1, 1],
				"wind_pressure": 200 * np.ones(NB_SPAN),
			}
		)
	)

	state_1 = sag_tension_calculation.change_state(weather_array_final)
	assert True
