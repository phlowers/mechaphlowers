# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0from abc import ABC, abstractmethod

import numpy as np
import pytest
from pandera.typing import pandas as pdt

from mechaphlowers.core.models.physics_based_cable_models import (
	PhysicsBasedCableModelImpl,
)
from mechaphlowers.core.models.space_position_cable_models import (
	CatenaryCableModel,
)
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


def test_physics_cable_impl(
	cable_array_input_data: dict,
) -> None:
	a = np.array([501.3, 499.0])
	b = np.array([0.0, -5.0])
	p = np.array([2_112.2, 2_112.0])
	lambd = np.array([16, 16.1])
	m = np.array([1, 1.1])

	cable_model = CatenaryCableModel(
		a, b, p, load_coefficient=m, linear_weight=lambd
	)

	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)
	physics_model = PhysicsBasedCableModelImpl(cable_model, input_df)
	current_temperature = np.array([20, 20])
	physics_model.L_ref(current_temperature)


# TODO: confirm values for this test
def test_physics_cable__two_spans(
	cable_array_input_data: dict,
) -> None:
	a = np.array([500, 500])
	b = np.array([0.0, 0.0])
	p = np.array([2_000, 2_000])
	m = np.array([1, 1])

	cable_model = CatenaryCableModel(a, b, p, load_coefficient=m)

	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)
	physics_model = PhysicsBasedCableModelImpl(cable_model, input_df)
	current_temperature = np.array([20, 20])
	physics_model.epsilon_mecha()
	physics_model.epsilon_therm(current_temperature)
	physics_model.epsilon(current_temperature)
	physics_model.L_ref(current_temperature)
