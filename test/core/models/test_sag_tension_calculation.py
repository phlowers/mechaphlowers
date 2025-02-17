# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
from pandera.typing import pandas as pdt

from mechaphlowers.core.models.cable.physics import Physics
from mechaphlowers.core.models.cable.span import (
	CatenarySpan,
)
from mechaphlowers.core.models.sag_tension_calculation import ScipySagTensionCalculation
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.entities.schemas import CableArrayInput


def test_functions_to_solve() -> None:
	input_cable: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		{
			"section": [345.55] * 4,
			"diameter": [22.4] * 4,
			"linear_weight": [9.55494] * 4,
			"young_modulus": [59] * 4,
			"dilatation_coefficient": [23] * 4,
			"temperature_reference": [0] * 4,
		}
	)
	data_section = {
		"name": ["support 1", "2", "three", "support 4"],
		"suspension": [False, True, True, False],
		"conductor_attachment_altitude": [2.2, 5, -0.12, 0],
		"crossarm_length": [10, 12.1, 10, 10.1],
		"line_angle": [0, 360, 90.1, -90.2],
		"insulator_length": [0, 4, 3.2, 0],
		"span_length": [1, 500.2, 500.0, np.nan],
	}

	section_array = SectionArray(data=pd.DataFrame(data_section))
	section_array.sagging_parameter = 2000
	section_array.sagging_temperature = 15
	
	a = np.array([500])
	b = np.array([0.0])
	p = np.array([2_000])
	m = np.array([1])
	linear_weight = np.array([9.55494])

	span_model = CatenarySpan(
		a, b, p, load_coefficient=m, linear_weight=linear_weight
	)
	cable_array = CableArray(input_cable)

	tension_mean = span_model.T_mean()
	cable_length = span_model.L()
	physics_model = Physics(cable_array, tension_mean, cable_length)
	unstressed_length = physics_model.L_ref(np.array([15] * 4))
	sag_tension_calculation = ScipySagTensionCalculation(section_array, unstressed_length, cable_array)
