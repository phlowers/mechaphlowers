# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.core.models.cable.deformation import DeformationRte
from mechaphlowers.core.models.cable.span import CatenarySpan
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.core.solver.cable_state import (
	SagTensionSolver,
)
from mechaphlowers.entities.data_container import DataContainer


@pytest.fixture
def weather_dict_one_span():
	return {
		"ice_thickness": 1e-2 * np.array([1.2, 3.4]),
		"wind_pressure": np.ones(2),
	}


def get_L_ref_from_arrays(
	data_container: DataContainer, current_temperature: np.ndarray
):
	cable_loads = CableLoads(**data_container.__dict__)
	span_model = CatenarySpan(**data_container.__dict__)

	span_model.load_coefficient = cable_loads.load_coefficient
	deformation = DeformationRte(
		**data_container.__dict__,
		tension_mean=span_model.T_mean(),
		cable_length=span_model.L(),
	)
	return deformation.L_ref()


def test_solver__run_solver(
	default_data_container_one_span: DataContainer,
	weather_dict_one_span: dict,
) -> None:
	current_temperature = np.array([15] * 2)

	sag_tension_calculation = SagTensionSolver(
		**default_data_container_one_span.__dict__
	)
	sag_tension_calculation.initial_state()

	sag_tension_calculation.change_state(
		**weather_dict_one_span,
		temp=current_temperature,
		solver="newton",
	)
	assert (
		sag_tension_calculation.cable_loads.ice_thickness
		== 1e-2 * np.array([1.2, 3.4])
	).all()
	assert (
		sag_tension_calculation.cable_loads.wind_pressure == np.ones(2)
	).all()
	# check no error
	sag_tension_calculation.p_after_change()
	sag_tension_calculation.L_after_change()


def test_solver__run_solver__polynomial_model(
	default_data_container_one_span: DataContainer,
	weather_dict_one_span: dict,
) -> None:
	current_temperature = np.array([15] * 2)
	# update polynomial from default data
	new_poly = Poly(
		[0, 1e9 * 100, 1e9 * -24_000, 1e9 * 2_440_000, 1e9 * -90_000_000]
	)
	default_data_container_one_span.polynomial_conductor = new_poly

	sag_tension_calculation = SagTensionSolver(
		**default_data_container_one_span.__dict__
	)
	sag_tension_calculation.initial_state()

	sag_tension_calculation.change_state(
		**weather_dict_one_span,
		temp=current_temperature,
		solver="newton",
	)
	# check no error
	sag_tension_calculation.p_after_change()
	sag_tension_calculation.L_after_change()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_solver__run_solver_no_solution(
	default_data_container_one_span: DataContainer,
	weather_dict_one_span: dict,
) -> None:
	current_temperature = np.array([15] * 2)
	sag_tension_calculation = SagTensionSolver(
		**default_data_container_one_span.__dict__
	)
	sag_tension_calculation.initial_state()
	sag_tension_calculation.L_ref = np.array([1, 1])
	with pytest.raises(ValueError) as excinfo:
		sag_tension_calculation.change_state(
			**weather_dict_one_span, temp=current_temperature
		)
	assert str(excinfo.value) == "Solver did not converge"


def test_solver__bad_solver(
	default_data_container_one_span: DataContainer,
	weather_dict_one_span: dict,
) -> None:
	current_temperature = np.array([15] * 2)

	sag_tension_calculation = SagTensionSolver(
		**default_data_container_one_span.__dict__
	)
	sag_tension_calculation.initial_state()
	with pytest.raises(ValueError) as excinfo:
		sag_tension_calculation.change_state(
			**weather_dict_one_span,
			temp=current_temperature,
			solver="wrong_solver",
		)
	assert str(excinfo.value) == "Incorrect solver name: wrong_solver"


def test_solver__values_before_solver(
	default_data_container_one_span: DataContainer,
) -> None:
	sag_tension_calculation = SagTensionSolver(
		**default_data_container_one_span.__dict__
	)

	assert sag_tension_calculation.T_h_after_change is None
	with pytest.raises(ValueError):
		sag_tension_calculation.p_after_change()
	with pytest.raises(ValueError):
		sag_tension_calculation.L_after_change()
