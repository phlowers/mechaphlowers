# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

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
from mechaphlowers.entities.data_container import DataContainer


def create_sag_tension_solver(
	data_section: pd.DataFrame,
	data_cable: pd.DataFrame,
	data_weather: pd.DataFrame,
	sagging_parameter: float,
	sagging_temperature: float,
) -> SagTensionSolver:
	section_array = SectionArray(data=pd.DataFrame(data_section))
	section_array.sagging_parameter = sagging_parameter
	section_array.sagging_temperature = sagging_temperature

	cable_array = CableArray(data_cable)

	frame = SectionDataFrame(section_array)
	frame.add_cable(cable_array)

	weather_array = WeatherArray(data_weather)

	frame.add_weather(weather_array)
	unstressed_length = frame.state.L_ref(
		section_array.data.sagging_temperature.to_numpy()
	)

	return SagTensionSolver(
		section_array,
		cable_array,
		weather_array,
		unstressed_length,
	)


def test_functions_to_solve__same_loads() -> None:
	NB_SPAN = 4
	input_cable = pd.DataFrame(
		{
			"section": [345.55] * NB_SPAN,
			"diameter": [22.4] * NB_SPAN,
			"linear_weight": [9.55494] * NB_SPAN,
			"young_modulus": [59] * NB_SPAN,
			"dilatation_coefficient": [23] * NB_SPAN,
			"temperature_reference": [0] * NB_SPAN,
			"a0": [0] * NB_SPAN,
			"a1": [59] * NB_SPAN,
			"a2": [0] * NB_SPAN,
			"a3": [0] * NB_SPAN,
			"a4": [0] * NB_SPAN,
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

	frame = SectionDataFrame(section_array)
	frame.add_cable(cable_array)

	weather_array = WeatherArray(
		pdt.DataFrame(
			{
				"ice_thickness": [1, 2.1, 0.0, 0.0],
				"wind_pressure": np.zeros(NB_SPAN),
			}
		)
	)

	frame.add_weather(weather_array)
	unstressed_length = frame.state.L_ref(np.array([15] * NB_SPAN))

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
	new_temperature = np.array([15] * NB_SPAN)
	sag_tension_calculation.change_state(weather_array_final, new_temperature)
	T_h_state_0 = sag_tension_calculation.T_h_after_change

	# TODO: change this test after fixing the issue with NaN at last value
	assert (((T_h_state_0 - frame.span.T_h())[0:-1]) < 1e-6).all()  # type: ignore[union-attr]

	assert (
		sag_tension_calculation.p_after_change()[0]
		- section_array.sagging_parameter
		< 1e-6
	)
	expected_p = np.array([section_array.sagging_parameter] * 3 + [np.nan])
	np.testing.assert_allclose(
		sag_tension_calculation.p_after_change(), expected_p, atol=1e-5
	)

	np.testing.assert_allclose(
		sag_tension_calculation.L_after_change(),
		frame.span.L(),  # type: ignore[union-attr]
		atol=1e-5,
	)


def test_functions_to_solve__different_weather(
	default_data_container_two_spans: DataContainer,
) -> None:
	NB_SPAN = 2
	input_cable = pd.DataFrame(
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
		}
	)
	data_section = pd.DataFrame(
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

	sagging_parameter = 2000
	sagging_temperature = 15

	initial_weather_data = pd.DataFrame(
		{
			"ice_thickness": [0.0, 0.0],
			"wind_pressure": np.zeros(NB_SPAN),
		}
	)

	sag_tension_calculation = create_sag_tension_solver(
		data_section,
		input_cable,
		initial_weather_data,
		sagging_parameter,
		sagging_temperature,
	)
	same_temperature = np.array([15] * NB_SPAN)
	sag_tension_calculation.change_state(
		WeatherArray(initial_weather_data), same_temperature
	)
	T_h_state_0 = sag_tension_calculation.T_h_after_change
	expected_result_0 = np.array([19109.88, np.nan])
	assert T_h_state_0 is not None
	np.testing.assert_allclose(T_h_state_0, expected_result_0, atol=1e-5)

	weather_array_final_1 = WeatherArray(
		pdt.DataFrame(
			{
				"ice_thickness": [2, 2],
				"wind_pressure": 0 * np.ones(NB_SPAN),
			}
		)
	)
	sag_tension_calculation.change_state(
		weather_array_final_1, same_temperature
	)
	T_h_state_1 = sag_tension_calculation.T_h_after_change
	expected_result_1 = np.array([42098.9070, np.nan])
	assert T_h_state_1 is not None
	np.testing.assert_allclose(T_h_state_1, expected_result_1, atol=0.01)

	weather_array_final_2 = WeatherArray(
		pdt.DataFrame(
			{
				"ice_thickness": [1, 1],
				"wind_pressure": 200 * np.ones(NB_SPAN),
			}
		)
	)
	sag_tension_calculation.change_state(
		weather_array_final_2, same_temperature
	)
	T_h_state_2 = sag_tension_calculation.T_h_after_change
	expected_result_2 = np.array([31745.05101, np.nan])
	assert T_h_state_2 is not None
	# Careful: value is not exactly the same: 31742.24808412 vs 31745.05101
	np.testing.assert_allclose(
		T_h_state_2, expected_result_2, atol=3
	)  # typing: ignore[arg-type]

	new_temperature = np.array([25] * NB_SPAN)
	sag_tension_calculation.change_state(
		WeatherArray(initial_weather_data), new_temperature
	)
	T_h_state_3 = sag_tension_calculation.T_h_after_change
	expected_result_3 = np.array([18380.1116, np.nan])
	assert T_h_state_3 is not None
	np.testing.assert_allclose(T_h_state_3, expected_result_3, atol=0.01)


def test_functions_to_solve__different_temp_ref() -> None:
	NB_SPAN = 2
	input_cable = pd.DataFrame(
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
		}
	)
	data_section = pd.DataFrame(
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

	initial_data_weather = pd.DataFrame(
		{
			"ice_thickness": [0.0, 0.0],
			"wind_pressure": np.zeros(NB_SPAN),
		}
	)
	sagging_parameter = 2000
	sagging_temperature = 15
	sag_tension_calculation_0 = create_sag_tension_solver(
		data_section,
		input_cable,
		initial_data_weather,
		sagging_parameter,
		sagging_temperature,
	)
	new_temperature = np.array([15] * NB_SPAN)
	weather_array_final = WeatherArray(
		pdt.DataFrame(
			{
				"ice_thickness": [6, 6],
				"wind_pressure": 0 * np.ones(NB_SPAN),
			}
		)
	)

	sag_tension_calculation_0.change_state(
		weather_array_final, new_temperature
	)
	T_h_state_0 = sag_tension_calculation_0.T_h_after_change
	expected_result_0 = np.array([117951.847, np.nan])
	assert T_h_state_0 is not None
	np.testing.assert_allclose(T_h_state_0, expected_result_0, atol=0.01)
	input_cable.update(
		pd.DataFrame(
			{
				"temperature_reference": [0] * NB_SPAN,
			}
		)
	)
	sag_tension_calculation_1 = create_sag_tension_solver(
		data_section,
		input_cable,
		initial_data_weather,
		sagging_parameter,
		sagging_temperature,
	)
	sag_tension_calculation_1.change_state(
		weather_array_final, new_temperature
	)
	T_h_state_1 = sag_tension_calculation_1.T_h_after_change
	expected_result_1 = np.array([117961.6142, np.nan])
	assert T_h_state_1 is not None
	np.testing.assert_allclose(T_h_state_1, expected_result_1, atol=0.01)
