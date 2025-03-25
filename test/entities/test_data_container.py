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
	factory_neutral_weather_array,
	default_cable_array,
):
	NB_SPAN = 4
	cable_dict = {
		"section": np.array([345.55] * NB_SPAN),
		"diameter": np.array([22.4] * NB_SPAN),
		"linear_weight": np.array([9.55494] * NB_SPAN),
		"young_modulus": np.array([59] * NB_SPAN),
		"dilatation_coefficient": np.array([23] * NB_SPAN),
		"temperature_reference": np.array([0] * NB_SPAN),
		"a0": np.array([0] * NB_SPAN),
		"a1": np.array([59] * NB_SPAN),
		"a2": np.array([0] * NB_SPAN),
		"a3": np.array([0] * NB_SPAN),
		"a4": np.array([0] * NB_SPAN),
		"b0": np.array([0] * NB_SPAN),
		"b1": np.array([0] * NB_SPAN),
		"b2": np.array([0] * NB_SPAN),
		"b3": np.array([0] * NB_SPAN),
		"b4": np.array([0] * NB_SPAN),
	}

	section_dict = {
		"name": np.array(["support 1", "2", "three", "support 4"]),
		"suspension": np.array([False, True, True, False]),
		"conductor_attachment_altitude": np.array([2.2, 5, -0.12, 0]),
		"crossarm_length": np.array([10, 12.1, 10, 10.1]),
		"line_angle": np.array([0, 360, 90.1, -90.2]),
		"insulator_length": np.array([0, 4, 3.2, 0]),
		"span_length": np.array([400, 500.2, 500.0, np.nan]),
	}

	weather_dict = {
		"ice_thickness": np.array([1, 2.1, 0.0, 0.0]),
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
	expected_result_poly = np.array(Poly(np.array([0, 59e9, 0, 0, 0])))
	expected_result_poly = np.repeat(expected_result_poly, 4)

	assert (
		data_container_new.young_modulus == cable_array.data["young_modulus"]
	).all()
	assert (
		data_container_new.crossarm_length
		== section_array.data["crossarm_length"]
	).all()
	assert (
		data_container_new.ice_thickness == weather_array.data["ice_thickness"]
	).all()
	assert (
		data_container_new.polynomial_conductor == expected_result_poly
	).all()


def test_factory_data_container__numpy_arrays():
	# same thing but check that the results are np.array even if input are python arrays
	NB_SPAN = 4
	cable_dict = {
		"section": [345.55] * NB_SPAN,
		"diameter": [22.4] * NB_SPAN,
		"linear_weight": [9.55494] * NB_SPAN,
		"young_modulus": [59] * NB_SPAN,
		"dilatation_coefficient": [23] * NB_SPAN,
		"temperature_reference": [0] * NB_SPAN,
		"a0": [0] * NB_SPAN,
		"a1": [59] * NB_SPAN,
		"a2": [0] * NB_SPAN,
		"a3": [0] * NB_SPAN,
		"a4": [0] * NB_SPAN,
		"b0": [0] * NB_SPAN,
		"b1": [0] * NB_SPAN,
		"b2": [0] * NB_SPAN,
		"b3": [0] * NB_SPAN,
		"b4": [0] * NB_SPAN,
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
