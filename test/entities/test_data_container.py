# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from mechaphlowers.entities.data_container import DataContainer
from mechaphlowers.entities.arrays import CableArray, SectionArray, WeatherArray
from mechaphlowers.entities.data_container import factory_data_container



def test_first_try():

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


	data_container_new = factory_data_container(section_array, cable_array, weather_array)
	data_container_new.name


	assert True
