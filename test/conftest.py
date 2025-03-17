# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.entities.arrays import (
	CableArray,
	SectionArray,
	WeatherArray,
)
from mechaphlowers.entities.data_container import (
	DataContainer,
	factory_data_container,
)

projet_dir: Path = Path(__file__).resolve().parents[1]
source_dir: Path = projet_dir / "src"
sys.path.insert(0, str(source_dir))


@pytest.fixture
def default_data_container_two_spans() -> DataContainer:
	NB_SPAN = 2
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
		"name": ["1", "2"],
		"suspension": [False, False],
		"conductor_attachment_altitude": [30, 40],
		"crossarm_length": [0, 0],
		"line_angle": [0, 0],
		"insulator_length": [0, 0],
		"span_length": [480, np.nan],
	}

	weather_dict = {
		"ice_thickness": [1, 2.1, 0.0, 0.0],
		"wind_pressure": np.zeros(NB_SPAN),
	}
	section_array = SectionArray(pd.DataFrame(section_dict))  # refactor this?
	section_array.sagging_parameter = 2000
	section_array.sagging_temperature = 15
	cable_array = CableArray(pd.DataFrame(cable_dict))
	weather_array = WeatherArray(pd.DataFrame(weather_dict))

	data_container_new = factory_data_container(
		section_array, cable_array, weather_array
	)
	return data_container_new
