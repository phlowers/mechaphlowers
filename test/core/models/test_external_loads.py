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

NB_SPAN = 3


# TODO: compare with actual values


# TODO: refacto: merge with fixture in test_physics_based_cable_model?
# in later issue?
@pytest.fixture
def cable() -> CableArray:
	# TODO: use np.full?
	cable_array_input_data = {
		"section": [345.5] * NB_SPAN,
		"diameter": [22.4] * NB_SPAN,
		"linear_weight": [9.6] * NB_SPAN,
		"young_modulus": [59] * NB_SPAN,
		"dilatation_coefficient": [23] * NB_SPAN,
		"temperature_reference": [15] * NB_SPAN,
	}
	cable_input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)
	return CableArray(cable_input_df)


def test_compute_ice_load(cable: CableArray) -> None:
	external_loads = ExternalLoads(
		cable,
		ice_thickness=np.array([0.01, 0.02, 0.0]),
		wind_pressure=np.array([0, 0]),
	)
	# TODO: improve API design?
	# external_loads = ExternalLoads(cable, ice_thickness=np.array([0.01, 0.02]))

	external_loads.ice_load()


def test_compute_wind_load(cable: CableArray) -> None:
	external_loads = ExternalLoads(
		cable,
		ice_thickness=np.array([0.01, 0.02, 0.0]),
		wind_pressure=np.array([240.12, 0, -240.13]),
	)
	# TODO: improve API design?
	# external_loads = ExternalLoads(cable, wind_pressure=np.array([240] * 2))

	external_loads.wind_load()


def test_total_load_coefficient_and_angle(cable: CableArray) -> None:
	external_loads = ExternalLoads(
		cable,
		ice_thickness=np.array(
			[
				0,
				0.01,
				0.02,
				0.0,
			]
		),
		wind_pressure=np.array(
			[
				0,
				240.12,
				0.0,
				240.13,
			]
		),
	)

	external_loads.total_load()
	external_loads.load_coefficient()
	external_loads.load_angle()
