# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.core.models.cable.deformation import DeformationRTE
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
from mechaphlowers.entities.data_container import DataContainer


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


@pytest.fixture
def neutral_weather_dict_one_span():
	return {
		"ice_thickness": np.array([0.0, 0.0]),
		"wind_pressure": np.zeros(2),
	}


def get_L_ref_from_arrays(
	data_container: DataContainer, current_temperature: np.ndarray
):
	cable_loads = CableLoads(**data_container.__dict__)
	span_model = CatenarySpan(**data_container.__dict__)

	span_model.load_coefficient = cable_loads.load_coefficient
	deformation = DeformationRTE(
		**data_container.__dict__,
		tension_mean=span_model.T_mean(),
		cable_length=span_model.L(),
	)
	return deformation.L_ref(current_temperature)


def test_solver__run_solver(
	default_data_container_two_spans: DataContainer,
	neutral_weather_dict_one_span: dict,
) -> None:
	current_temperature = np.array([15] * 2)
	unstressed_length = get_L_ref_from_arrays(
		default_data_container_two_spans,
		current_temperature,
	)

	sag_tension_calculation = SagTensionSolver(
		**default_data_container_two_spans.__dict__,
		unstressed_length=unstressed_length,
	)
	sag_tension_calculation.change_state(
		**neutral_weather_dict_one_span,
		temp=current_temperature,
		solver="newton",
	)
	# check no error
	sag_tension_calculation.p_after_change()
	sag_tension_calculation.L_after_change()


def test_solver__run_solver__polynomial_model(
	default_data_container_two_spans: DataContainer,
	neutral_weather_dict_one_span: dict,
) -> None:
	current_temperature = np.array([15] * 2)
	# update polynomial from default data
	new_poly = Poly(
		[0, 1e9 * 100, 1e9 * -24_000, 1e9 * 2_440_000, 1e9 * -90_000_000]
	)
	new_poly_array = np.repeat(np.array([new_poly]), 2)
	default_data_container_two_spans.polynomial_conductor = new_poly_array

	unstressed_length = get_L_ref_from_arrays(
		default_data_container_two_spans,
		current_temperature,
	)

	sag_tension_calculation = SagTensionSolver(
		**default_data_container_two_spans.__dict__,
		unstressed_length=unstressed_length,
	)
	sag_tension_calculation.change_state(
		**neutral_weather_dict_one_span,
		temp=current_temperature,
		solver="newton",
	)
	# check no error
	sag_tension_calculation.p_after_change()
	sag_tension_calculation.L_after_change()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_solver__run_solver_no_solution(
	default_data_container_two_spans: DataContainer,
	neutral_weather_dict_one_span: dict,
) -> None:
	current_temperature = np.array([15] * 2)
	sag_tension_calculation = SagTensionSolver(
		**default_data_container_two_spans.__dict__,
		unstressed_length=np.array([1, 1]),
	)
	with pytest.raises(ValueError) as excinfo:
		sag_tension_calculation.change_state(
			**neutral_weather_dict_one_span, temp=current_temperature
		)
	assert str(excinfo.value) == "Solver did not converge"


def test_solver__bad_solver(
	default_data_container_two_spans: DataContainer,
	neutral_weather_dict_one_span: dict,
) -> None:
	current_temperature = np.array([15] * 2)
	unstressed_length = get_L_ref_from_arrays(
		default_data_container_two_spans,
		current_temperature,
	)

	sag_tension_calculation = SagTensionSolver(
		**default_data_container_two_spans.__dict__,
		unstressed_length=unstressed_length,
	)

	with pytest.raises(ValueError) as excinfo:
		sag_tension_calculation.change_state(
			**neutral_weather_dict_one_span,
			temp=current_temperature,
			solver="wrong_solver",
		)
	assert str(excinfo.value) == "Incorrect solver name: wrong_solver"


def test_solver__values_before_solver(
	default_data_container_two_spans: DataContainer,
	neutral_weather_dict_one_span: dict,
) -> None:
	current_temperature = np.array([15] * 2)
	unstressed_length = get_L_ref_from_arrays(
		default_data_container_two_spans,
		current_temperature,
	)

	sag_tension_calculation = SagTensionSolver(
		**default_data_container_two_spans.__dict__,
		unstressed_length=unstressed_length,
	)
	assert sag_tension_calculation.T_h_after_change is None
	with pytest.raises(ValueError):
		sag_tension_calculation.p_after_change()
	with pytest.raises(ValueError):
		sag_tension_calculation.L_after_change()
