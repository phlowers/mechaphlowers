# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from pandera.typing import pandas as pdt

from mechaphlowers.core.models.cable.deformation import DeformationImpl
from mechaphlowers.core.models.cable.span import (
	CatenarySpan,
)
from mechaphlowers.entities.arrays import CableArray
from mechaphlowers.entities.schemas import CableArrayInput


@pytest.fixture
def cable_array_input_data() -> dict[str, list]:
	return {
		"section": [345.5, 345.5],
		"diameter": [22.4, 22.4],
		"linear_weight": [9.6, 9.6],
		"young_modulus": [59, 59],
		"dilatation_coefficient": [23, 23],
		"temperature_reference": [15, 15],
		"a0": [0] * 2,
		"a1": [59] * 2,
		"a2": [0] * 2,
		"a3": [0] * 2,
		"a4": [0] * 2,
	}


@pytest.fixture
def a_two_spans() -> np.ndarray:
	return np.array([500, 500])


@pytest.fixture
def b_two_spans() -> np.ndarray:
	return np.array([0.0, -5.0])


@pytest.fixture
def p_two_spans() -> np.ndarray:
	return np.array([2_000, 2_000.0])


@pytest.fixture
def lambd_two_spans() -> np.ndarray:
	return np.array([9.6, 9.6])


@pytest.fixture
def m_two_spans() -> np.ndarray:
	return np.array([1, 1])


def test_elastic_linear_cable_impl(
	cable_array_input_data: dict,
) -> None:
	a = np.array([501.3, 499.0])
	b = np.array([0.0, -5.0])
	p = np.array([2_112.2, 2_112.0])
	lambd = np.array([9.6, 9.6])
	m = np.array([1, 1.1])

	span_model = CatenarySpan(a, b, p, load_coefficient=m, linear_weight=lambd)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()

	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)

	cable_array = CableArray(input_df)
	linear_model = DeformationImpl(cable_array, tension_mean, cable_length)
	current_temperature = np.array([15, 15])
	linear_model.epsilon_mecha()
	linear_model.epsilon_therm(current_temperature)
	linear_model.epsilon(current_temperature)
	linear_model.L_ref(current_temperature)


def test_deformation_values__first_example() -> None:
	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
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

	a = np.array([500] * 2)
	b = np.array([0.0] * 2)
	p = np.array([2_000] * 2)
	m = np.array([1] * 2)
	linear_weight = np.array([9.55494] * 2)

	span_model = CatenarySpan(
		a, b, p, load_coefficient=m, linear_weight=linear_weight
	)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()
	cable_array = CableArray(input_df)
	linear_model = DeformationImpl(cable_array, tension_mean, cable_length)
	current_temperature = np.array([15, 15])

	# Data given by the prototype
	eps_mecha = linear_model.epsilon_mecha()
	eps_therm = linear_model.epsilon_therm(current_temperature)
	L_ref = linear_model.L_ref(current_temperature)

	np.testing.assert_allclose(
		eps_mecha,
		np.array([0.00093978, 0.00093978]),
		atol=1e-6,
	)
	np.testing.assert_allclose(
		eps_therm,
		np.array([0.000345, 0.000345]),
		atol=1e-6,
	)
	np.testing.assert_allclose(
		L_ref,
		np.array([500.65986147, 500.65986147]),
		atol=1e-6,
	)


def test_poly_deformation__degree_one(
	cable_array_input_data: dict,
	a_two_spans: np.ndarray,
	b_two_spans: np.ndarray,
	p_two_spans: np.ndarray,
	lambd_two_spans: np.ndarray,
	m_two_spans: np.ndarray,
) -> None:
	span_model = CatenarySpan(
		a_two_spans,
		b_two_spans,
		p_two_spans,
		load_coefficient=m_two_spans,
		linear_weight=lambd_two_spans,
	)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()

	cable_array_input_data.update(
		{
			"a0": [0] * 2,
			"a1": [26.760] * 2,
			"a2": [0] * 2,
			"a3": [0] * 2,
			"a4": [0] * 2,
		}
	)

	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)

	cable_array = CableArray(input_df)
	polynomial_deformation_model = DeformationImpl(
		cable_array, tension_mean, cable_length
	)
	constraint = tension_mean / (
		np.array(cable_array_input_data["section"]) * 1e-6
	)
	current_temperature = np.array([15, 15])
	polynomial_deformation_model.resolve_stress_strain_equation(
		constraint, cable_array.stress_strain_polynomial
	)
	polynomial_deformation_model.epsilon_mecha()
	polynomial_deformation_model.epsilon(current_temperature)


def test_poly_deformation__degree_three(
	cable_array_input_data: dict,
	a_two_spans: np.ndarray,
	b_two_spans: np.ndarray,
	p_two_spans: np.ndarray,
	lambd_two_spans: np.ndarray,
	m_two_spans: np.ndarray,
) -> None:
	span_model = CatenarySpan(
		a_two_spans,
		b_two_spans,
		p_two_spans,
		load_coefficient=m_two_spans,
		linear_weight=lambd_two_spans,
	)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()

	cable_array_input_data.update(
		{
			"a0": [0] * 2,
			"a1": [50] * 2,
			"a2": [-3_000] * 2,
			"a3": [44_000] * 2,
			"a4": [0] * 2,
		}
	)

	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)

	cable_array = CableArray(input_df)
	polynomial_deformation_model = DeformationImpl(
		cable_array, tension_mean, cable_length
	)
	constraint = tension_mean / (
		np.array(cable_array_input_data["section"]) * 1e-6
	)
	current_temperature = np.array([15, 15])
	polynomial_deformation_model.resolve_stress_strain_equation(
		constraint, cable_array.stress_strain_polynomial
	)
	polynomial_deformation_model.epsilon_mecha()

	polynomial_deformation_model.epsilon(current_temperature)


def test_poly_deformation__degree_four(
	cable_array_input_data: dict,
	a_two_spans: np.ndarray,
	b_two_spans: np.ndarray,
	p_two_spans: np.ndarray,
	lambd_two_spans: np.ndarray,
	m_two_spans: np.ndarray,
) -> None:
	span_model = CatenarySpan(
		a_two_spans,
		b_two_spans,
		p_two_spans,
		load_coefficient=m_two_spans,
		linear_weight=lambd_two_spans,
	)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()

	cable_array_input_data.update(
		{
			"a0": [0] * 2,
			"a1": [100] * 2,
			"a2": [-24_000] * 2,
			"a3": [2_440_000] * 2,
			"a4": [-90_000_000] * 2,
		}
	)

	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)

	cable_array = CableArray(input_df)
	polynomial_deformation_model = DeformationImpl(
		cable_array, tension_mean, cable_length
	)
	constraint = tension_mean / (
		np.array(cable_array_input_data["section"]) * 1e-6
	)
	current_temperature = np.array([15, 15])
	polynomial_deformation_model.resolve_stress_strain_equation(
		constraint, cable_array.stress_strain_polynomial
	)
	polynomial_deformation_model.epsilon_mecha()

	polynomial_deformation_model.epsilon(current_temperature)


def test_poly_deformation__degree_four__with_max_stress(
	cable_array_input_data: dict,
	a_two_spans: np.ndarray,
	b_two_spans: np.ndarray,
	p_two_spans: np.ndarray,
	lambd_two_spans: np.ndarray,
	m_two_spans: np.ndarray,
) -> None:
	span_model = CatenarySpan(
		a_two_spans,
		b_two_spans,
		p_two_spans,
		load_coefficient=m_two_spans,
		linear_weight=lambd_two_spans,
	)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()

	cable_array_input_data.update(
		{
			"a0": [0] * 2,
			"a1": [100] * 2,
			"a2": [-24_000] * 2,
			"a3": [2_440_000] * 2,
			"a4": [-90_000_000] * 2,
		}
	)

	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)

	cable_array = CableArray(input_df)
	default_max_stress = np.array([0, 0])
	polynomial_deformation_model = DeformationImpl(
		cable_array, tension_mean, default_max_stress, cable_length
	)

	current_temperature = np.array([15, 15])
	polynomial_deformation_model.max_stress = np.array([1000, 1e8])
	polynomial_deformation_model.epsilon_mecha()
	polynomial_deformation_model.epsilon(current_temperature)


def test_poly_deformation__no_solutions(
	cable_array_input_data: dict,
	a_two_spans: np.ndarray,
	b_two_spans: np.ndarray,
	p_two_spans: np.ndarray,
	lambd_two_spans: np.ndarray,
	m_two_spans: np.ndarray,
) -> None:
	span_model = CatenarySpan(
		a_two_spans,
		b_two_spans,
		p_two_spans,
		load_coefficient=m_two_spans,
		linear_weight=lambd_two_spans,
	)
	tension_mean = span_model.T_mean()
	cable_length = span_model.L()

	cable_array_input_data.update(
		{
			"a0": [0] * 2,
			"a1": [100] * 2,
			"a2": [-24_000] * 2,
			"a3": [2_440_000] * 2,
			"a4": [-90_000_000] * 2,
		}
	)

	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)

	cable_array = CableArray(input_df)
	polynomial_deformation_model = DeformationImpl(
		cable_array, tension_mean, cable_length
	)
	polynomial_deformation_model.max_stress = np.array([1000, 1e10])
	with pytest.raises(ValueError):
		polynomial_deformation_model.epsilon_mecha()
