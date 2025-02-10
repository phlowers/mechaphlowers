# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from pandera.typing import pandas as pdt

from mechaphlowers.core.models.cable.deformation import (
	LinearDeformation,
	PolynomialDeformation,
)
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
	}


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

	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)

	cable_array = CableArray(input_df)
	linear_model = LinearDeformation(cable_array, tension_mean)
	current_temperature = np.array([15, 15])
	linear_model.epsilon_mecha()
	linear_model.epsilon_therm(current_temperature)
	linear_model.epsilon(current_temperature)


def test_physics_cable__first_example() -> None:
	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		{
			"section": [345.55],
			"diameter": [22.4],
			"linear_weight": [9.55494],
			"young_modulus": [59],
			"dilatation_coefficient": [23],
			"temperature_reference": [0],
		}
	)

	a = np.array([500])
	b = np.array([0.0])
	p = np.array([2_000])
	m = np.array([1])
	linear_weight = np.array([9.55494])

	span_model = CatenarySpan(
		a, b, p, load_coefficient=m, linear_weight=linear_weight
	)
	tension_mean = span_model.T_mean()
	cable_array = CableArray(input_df)
	linear_model = LinearDeformation(cable_array, tension_mean)
	current_temperature = np.array([15])

	# Data given by the prototype
	assert abs(linear_model.epsilon_mecha() - 0.00093978) < 0.01
	assert (
		abs(linear_model.epsilon_therm(current_temperature) + 0.000345) < 0.01
	)


def test_poly_deformation__degree_one(
	cable_array_input_data: dict,
) -> None:
	a = np.array([500, 500])
	b = np.array([0.0, -5.0])
	p = np.array([2_000, 2_000.0])
	lambd = np.array([9.6, 9.6])
	m = np.array([1, 1])

	span_model = CatenarySpan(a, b, p, load_coefficient=m, linear_weight=lambd)
	tension_mean = span_model.T_mean()

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
	polynomial_deformation_model = PolynomialDeformation(
		cable_array, tension_mean
	)
	constraint = tension_mean / (
		np.array(cable_array_input_data["section"]) * 1e-6
	)
	current_temperature = np.array([15, 15])
	polynomial_deformation_model.find_roots_polynom(constraint)
	polynomial_deformation_model.epsilon_mecha(np.array([1000, 1e8]))
	polynomial_deformation_model.epsilon(current_temperature)


def test_poly_deformation__degree_three(
	cable_array_input_data: dict,
) -> None:
	a = np.array([500, 500])
	b = np.array([0.0, -5.0])
	p = np.array([2_000, 2_000.0])
	lambd = np.array([9.6, 9.6])
	m = np.array([1, 1])

	span_model = CatenarySpan(a, b, p, load_coefficient=m, linear_weight=lambd)
	tension_mean = span_model.T_mean()

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
	polynomial_deformation_model = PolynomialDeformation(
		cable_array, tension_mean
	)
	constraint = tension_mean / (
		np.array(cable_array_input_data["section"]) * 1e-6
	)
	current_temperature = np.array([15, 15])
	polynomial_deformation_model.find_roots_polynom(constraint)
	polynomial_deformation_model.epsilon_mecha(np.array([1000, 1e8]))

	polynomial_deformation_model.epsilon(current_temperature)


def test_poly_deformation__degree_four(
	cable_array_input_data: dict,
) -> None:
	a = np.array([500, 500])
	b = np.array([0.0, -5.0])
	p = np.array([2_000, 2_000.0])
	lambd = np.array([9.6, 9.6])
	m = np.array([1, 1])

	span_model = CatenarySpan(a, b, p, load_coefficient=m, linear_weight=lambd)
	tension_mean = span_model.T_mean()

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
	polynomial_deformation_model = PolynomialDeformation(
		cable_array, tension_mean
	)
	constraint = tension_mean / (
		np.array(cable_array_input_data["section"]) * 1e-6
	)
	current_temperature = np.array([15, 15])
	polynomial_deformation_model.find_roots_polynom(constraint)
	polynomial_deformation_model.epsilon_mecha(np.array([1000, 1e8]))

	polynomial_deformation_model.epsilon(current_temperature)

def test_poly_deformation__no_solutions(
	cable_array_input_data: dict,
) -> None:
	a = np.array([500, 500])
	b = np.array([0.0, -5.0])
	p = np.array([2_000, 2_000.0])
	lambd = np.array([9.6, 9.6])
	m = np.array([1, 1])

	span_model = CatenarySpan(a, b, p, load_coefficient=m, linear_weight=lambd)
	tension_mean = span_model.T_mean()

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
	polynomial_deformation_model = PolynomialDeformation(
		cable_array, tension_mean
	)
	with pytest.raises(ValueError):
		polynomial_deformation_model.epsilon_mecha(np.array([1000, 1e10]))
