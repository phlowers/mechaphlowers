# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.entities.arrays import (
	CableArray,
	SectionArray,
	WeatherArray,
)
from mechaphlowers.entities.data_container import factory_data_container


def test_factory_data_container(
	default_section_array_four_spans: SectionArray,
	default_cable_array,
	factory_neutral_weather_array,
):
	NB_SPAN = 4

	weather_array = factory_neutral_weather_array(NB_SPAN)

	data_container_new = factory_data_container(
		default_section_array_four_spans, default_cable_array, weather_array
	)
	expected_result_poly = Poly(np.array([0, 59e9, 0, 0, 0]))

	assert (
		data_container_new.young_modulus
		== default_cable_array.data["young_modulus"][0]
	)
	assert (
		data_container_new.crossarm_length
		== default_section_array_four_spans.data["crossarm_length"]
	).all()
	assert (
		data_container_new.ice_thickness == weather_array.data["ice_thickness"]
	).all()
	assert data_container_new.polynomial_conductor == expected_result_poly


def test_factory_data_container__numpy_arrays():
	# same thing but check that the results are np.array even if input are python arrays
	NB_SPAN = 4
	cable_dict = {
		"section": [345.55],
		"diameter": [22.4],
		"linear_weight": [9.55494],
		"young_modulus": [59],
		"dilatation_coefficient": [23],
		"temperature_reference": [0],
		"a0": [0],
		"a1": [59],
		"a2": [0],
		"a3": [0],
		"a4": [0],
		"b0": [0],
		"b1": [0],
		"b2": [0],
		"b3": [0],
		"b4": [0],
	}

	section_dict = {
		"name": ["support 1", "2", "three", "support 4"],
		"suspension": [False, True, True, False],
		"conductor_attachment_altitude": [2.2, 5, -0.12, 0],
		"crossarm_length": [10, 12.1, 10, 10.1],
		"line_angle": [0, 360, 90.1, -90.2],
		"insulator_length": [0, 4, 3.2, 0],
		"span_length": [400, 500.2, 500.0, np.nan],
	}

	weather_dict = {
		"ice_thickness": [1, 2.1, 0.0, 0.0],
		"wind_pressure": np.zeros(NB_SPAN),
	}

	section_array = SectionArray(pd.DataFrame(section_dict))
	section_array.sagging_parameter = 2000
	section_array.sagging_temperature = 15
	cable_array = CableArray(pd.DataFrame(cable_dict))
	weather_array = WeatherArray(pd.DataFrame(weather_dict))

	data_container_new = factory_data_container(
		section_array, cable_array, weather_array
	)
	data_container_new.support_name

	assert True


# TODO: test about factory

# TODO: test about add_cable/add_weather...
