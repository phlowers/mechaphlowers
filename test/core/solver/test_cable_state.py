# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.models.cable.deformation import DeformationImpl
from mechaphlowers.core.models.cable.span import CatenarySpan
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.core.solver.cable_state import (
	SagTensionSolver,
)
from mechaphlowers.entities.arrays import (
	CableArray,
	SectionArray,
	WeatherArray,
)


@pytest.fixture
def section_array_one_span():
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
			}
		)
	)
	section_array.sagging_parameter = 2000
	section_array.sagging_temperature = 15
	return section_array


@pytest.fixture
def cable_array_one_span():
	return CableArray(
		pd.DataFrame(
			{
				"section": [345.55] * 2,
				"diameter": [22.4] * 2,
				"linear_weight": [9.55494] * 2,
				"young_modulus": [59] * 2,
				"dilatation_coefficient": [23] * 2,
				"temperature_reference": [0] * 2,
				"a0": [0] * 2,
				"a1": [59] * 2,
				"a2": [0] * 2,
				"a3": [0] * 2,
				"a4": [0] * 2,
			}
		)
	)


@pytest.fixture
def cable_array_one_span__polynomial():
	return CableArray(
		pd.DataFrame(
			{
				"section": [345.55] * 2,
				"diameter": [22.4] * 2,
				"linear_weight": [9.55494] * 2,
				"young_modulus": [59] * 2,
				"dilatation_coefficient": [23] * 2,
				"temperature_reference": [0] * 2,
				"a0": [0] * 2,
				"a1": [100] * 2,
				"a2": [-24_000] * 2,
				"a3": [2_440_000] * 2,
				"a4": [-90_000_000] * 2,
			}
		)
	)


@pytest.fixture
def neutral_weather_array_one_span():
	return WeatherArray(
		pd.DataFrame(
			{
				"ice_thickness": [0.0, 0.0],
				"wind_pressure": np.zeros(2),
			}
		)
	)


def get_L_ref_from_arrays(
	section_array: SectionArray,
	cable_array: CableArray,
	weather_array: WeatherArray,
	current_temperature: np.ndarray,
):
	cable_loads = CableLoads(cable_array, weather_array)
	span_model = CatenarySpan(
		section_array.data.span_length.to_numpy(),
		section_array.data.elevation_difference.to_numpy(),
		section_array.data.sagging_parameter.to_numpy(),
	)
	span_model.load_coefficient = cable_loads.load_coefficient
	span_model.linear_weight = cable_array.data.linear_weight.to_numpy()
	deformation = DeformationImpl(
		cable_array,
		span_model.T_mean(),
		span_model.L(),
	)
	return deformation.L_ref(current_temperature)


def test_solver__run_solver(
	section_array_one_span: SectionArray,
	cable_array_one_span: CableArray,
	neutral_weather_array_one_span: WeatherArray,
) -> None:
	current_temperature = np.array([15] * 2)
	unstressed_length = get_L_ref_from_arrays(
		section_array_one_span,
		cable_array_one_span,
		neutral_weather_array_one_span,
		current_temperature,
	)

	sag_tension_calculation = SagTensionSolver(
		section_array_one_span,
		cable_array_one_span,
		neutral_weather_array_one_span,
		unstressed_length,
	)
	sag_tension_calculation.change_state(
		neutral_weather_array_one_span, current_temperature, "newton"
	)
	sag_tension_calculation.p_after_change()
	sag_tension_calculation.L_after_change()


def test_solver__run_solver__polynomial_model(
	section_array_one_span: SectionArray,
	cable_array_one_span__polynomial: CableArray,
	neutral_weather_array_one_span: WeatherArray,
) -> None:
	current_temperature = np.array([15] * 2)
	# add polynomial
	unstressed_length = get_L_ref_from_arrays(
		section_array_one_span,
		cable_array_one_span__polynomial,
		neutral_weather_array_one_span,
		current_temperature,
	)

	sag_tension_calculation = SagTensionSolver(
		section_array_one_span,
		cable_array_one_span__polynomial,
		neutral_weather_array_one_span,
		unstressed_length,
	)
	sag_tension_calculation.change_state(
		neutral_weather_array_one_span, current_temperature, "newton"
	)
	sag_tension_calculation.p_after_change()
	sag_tension_calculation.L_after_change()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_solver__run_solver_no_solution(
	section_array_one_span: SectionArray,
	cable_array_one_span: CableArray,
	neutral_weather_array_one_span: WeatherArray,
) -> None:
	current_temperature = np.array([15] * 2)
	sag_tension_calculation = SagTensionSolver(
		section_array_one_span,
		cable_array_one_span,
		neutral_weather_array_one_span,
		np.array([1, 1]),
	)
	with pytest.raises(ValueError) as excinfo:
		sag_tension_calculation.change_state(
			neutral_weather_array_one_span, current_temperature
		)
	assert str(excinfo.value) == "Solver did not converge"


def test_solver__bad_solver(
	section_array_one_span: SectionArray,
	cable_array_one_span: CableArray,
	neutral_weather_array_one_span: WeatherArray,
) -> None:
	current_temperature = np.array([15] * 2)
	unstressed_length = get_L_ref_from_arrays(
		section_array_one_span,
		cable_array_one_span,
		neutral_weather_array_one_span,
		current_temperature,
	)

	sag_tension_calculation = SagTensionSolver(
		section_array_one_span,
		cable_array_one_span,
		neutral_weather_array_one_span,
		unstressed_length,
	)
	with pytest.raises(ValueError) as excinfo:
		sag_tension_calculation.change_state(
			neutral_weather_array_one_span, current_temperature, "wrong_solver"
		)
	assert str(excinfo.value) == "Incorrect solver name: wrong_solver"


def test_solver__values_before_solver(
	section_array_one_span: SectionArray,
	cable_array_one_span: CableArray,
	neutral_weather_array_one_span: WeatherArray,
) -> None:
	current_temperature = np.array([15] * 2)
	unstressed_length = get_L_ref_from_arrays(
		section_array_one_span,
		cable_array_one_span,
		neutral_weather_array_one_span,
		current_temperature,
	)

	sag_tension_calculation = SagTensionSolver(
		section_array_one_span,
		cable_array_one_span,
		neutral_weather_array_one_span,
		unstressed_length,
	)
	assert sag_tension_calculation.T_h_after_change is None
	with pytest.raises(ValueError):
		sag_tension_calculation.p_after_change()
	with pytest.raises(ValueError):
		sag_tension_calculation.L_after_change()
