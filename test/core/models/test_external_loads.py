# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from pandera.typing import pandas as pdt

from mechaphlowers.core.models.external_loads import WeatherLoads
from mechaphlowers.entities.arrays import CableArray
from mechaphlowers.entities.schemas import CableArrayInput

NB_SPAN = 3


# TODO: compare with actual values from prototype?


# TODO: refacto: merge with fixture in test_physics_based_cable_model?
# in later issue?
@pytest.fixture
def cable() -> CableArray:
	cable_array_input_data = {
		"section": np.full(NB_SPAN, [345.5]),
		"diameter": np.full(NB_SPAN, [22.4]),
		"linear_weight": np.full(NB_SPAN, [9.6]),
		"young_modulus": np.full(NB_SPAN, [59]),
		"dilatation_coefficient": np.full(NB_SPAN, [23]),
		"temperature_reference": np.full(NB_SPAN, [15]),
	}
	cable_input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)
	return CableArray(cable_input_df)


def test_compute_ice_load(cable: CableArray) -> None:
	external_loads = WeatherLoads(
		cable,
		ice_thickness=np.array([0.01, 0.02, 0.0]),
		wind_pressure=np.array([0, 0]),
	)
	# TODO: improve API design?
	# external_loads = WeatherLoads(cable, ice_thickness=np.array([0.01, 0.02]))

	external_loads.ice_load()


def test_compute_wind_load(cable: CableArray) -> None:
	external_loads = WeatherLoads(
		cable,
		ice_thickness=np.array([0.01, 0.02, 0.0]),
		wind_pressure=np.array([240.12, 0, -240.13]),
	)
	# TODO: improve API design?
	# external_loads = WeatherLoads(cable, wind_pressure=np.array([240] * 2))

	external_loads.wind_load()


def test_total_load_coefficient_and_angle(cable: CableArray) -> None:
	external_loads = WeatherLoads(
		cable,
		ice_thickness=np.array(
			[
				0.01,
				0.02,
				0.0,
			]
		),
		wind_pressure=np.array(
			[
				240.12,
				0.0,
				240.13,
			]
		),
	)

	result = external_loads.result()
	result.load_coefficient
	result.load_angle
