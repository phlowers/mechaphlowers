# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0from abc import ABC, abstractmethod

import numpy as np
import pytest
from pandera.typing import pandas as pdt

from mechaphlowers.core.models.mecha_cable_extension_models import (
	ElasticLinearExtensionModel,
	ElasticPolynomialExtensionModel,
)
from mechaphlowers.core.models.space_position_cable_models import (
	CatenaryCableModel,
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

	cable_model = CatenaryCableModel(
		a, b, p, load_coefficient=m, linear_weight=lambd
	)

	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)

	cable_array = CableArray(input_df)

	linear_mecha_model = ElasticLinearExtensionModel(cable_model, cable_array)
	linear_mecha_model.epsilon_mecha()


def test_poly_extension_cable_impl(
	cable_array_input_data: dict,
) -> None:
	a = np.array([501.3, 499.0])
	b = np.array([0.0, -5.0])
	p = np.array([2_112.2, 2_112.0])
	lambd = np.array([9.6, 9.6])
	m = np.array([1, 1])

	cable_model = CatenaryCableModel(
		a, b, p, load_coefficient=m, linear_weight=lambd
	)

	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)
	coefs_poly_heart = np.array([0, 38_440])
	coefs_poly_conductor = np.array(
		[0, 26_760, -6344000, 646900000, -23640000000]
	)

	cable_array = CableArray(input_df, coefs_poly_heart, coefs_poly_conductor)
	polynomial_mecha_model = ElasticPolynomialExtensionModel(
		cable_model, cable_array
	)
	# Test not working: returns no solutions
	polynomial_mecha_model.epsilon_mecha()


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

	cable_model = CatenaryCableModel(
		a, b, p, load_coefficient=m, linear_weight=linear_weight
	)

	cable_array = CableArray(input_df, np.array([0]), np.array([0]))
	mecha_model = ElasticLinearExtensionModel(cable_model, cable_array)

	# Data given by the prototype
	assert abs(mecha_model.epsilon_mecha() - 0.00093978) < 0.01
