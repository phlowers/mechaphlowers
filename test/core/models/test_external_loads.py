# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from pandera.typing import pandas as pdt

from mechaphlowers.core.models.external_loads import ExternalLoads
from mechaphlowers.entities.arrays import CableArray
from mechaphlowers.entities.schemas import CableArrayInput


# TODO: refacto: merge with fixture in test_physics_based_cable_model?
# in later issue?
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


def test_compute_ice_load(cable_array_input_data: dict[str, list]) -> None:
	cable_input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)
	cable = CableArray(cable_input_df)

	external_loads = ExternalLoads(
		cable,
		ice_thickness=np.array([0.01, 0.02]),
		wind_pressure=np.array([0, 0]),
	)
	# TODO: improve API design?
	# external_loads = ExternalLoads(cable, ice_thickness=np.array([0.01, 0.02]))

	external_loads.ice_load()  # TODO: compare with actual value


def test_compute_wind_load(cable_array_input_data: dict[str, list]) -> None:
	cable_input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)
	cable = CableArray(cable_input_df)

	external_loads = ExternalLoads(
		cable,
		ice_thickness=np.array([0, 0]),
		wind_pressure=np.array([240, -240]),
	)
	# TODO: improve API design?
	# external_loads = ExternalLoads(cable, wind_pressure=np.array([240] * 2))

	external_loads.wind_load()  # TODO: compare with actual value
