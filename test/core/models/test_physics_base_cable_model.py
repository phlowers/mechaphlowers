# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0from abc import ABC, abstractmethod

import numpy as np
import pytest
from pandera.typing import pandas as pdt

from mechaphlowers.core.models.physics_based_cable_model import (
	PhysicsBasedCableModelImpl,
)
from mechaphlowers.core.models.space_position_cable_models import (
	CatenaryCableModel,
)
from mechaphlowers.entities.schemas import CableArrayInput


@pytest.fixture
def section_array_input_data() -> dict[str, list]:
	return {
		"name": ["support 1", "2", "three", "support 4"],
		"suspension": [False, True, True, False],
		"conductor_attachment_altitude": [2.2, 5, -0.12, 0],
		"crossarm_length": [10, 12.1, 10, 10.1],
		"line_angle": [0, 360, 90.1, -90.2],
		"insulator_length": [0, 4, 3.2, 0],
		"span_length": [1, 500.2, 500.05, np.nan],
	}


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
	section_array_input_data: dict,
	cable_array_input_data: dict,
) -> None:
	a = np.array([501.3, 499.0])
	b = np.array([0.0, -5.0])
	p = np.array([2_112.2, 2_112.0])
	lambd = np.array([16, 16.1])
	m = np.array([1, 1.1])

	cable_model = CatenaryCableModel(a, b, p, lambd, m)

	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)
	physics_model = PhysicsBasedCableModelImpl(cable_model, input_df)
	current_temperature = np.array([20, 20])
	physics_model.L_ref(current_temperature)
