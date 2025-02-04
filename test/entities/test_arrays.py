# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pandera as pa
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from pandera.typing import pandas as pdt

from mechaphlowers.entities.arrays import (
	CableArray,
	CableArrayInput,
	SectionArray,
	SectionArrayInput,
)


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
def section_array(section_array_input_data: dict[str, list]) -> SectionArray:
	df = pdt.DataFrame[SectionArrayInput](section_array_input_data)
	return SectionArray(
		data=df, sagging_parameter=2_000, sagging_temperature=15
	)


@pytest.fixture
def cable_array_input_data() -> dict[str, list]:
	return {
		"section": [345.5, 345.5, 345.5, 345.5],
		"diameter": [22.4, 22.4, 22.4, 22.4],
		"linear_weight": [9.6, 9.6, 9.6, 9.6],
		"young_modulus": [59, 59, 59, 59],
		"dilatation_coefficient": [23, 23, 23, 23],
		"temperature_reference": [15, 15, 15, 15],
	}


def test_create_section_array__with_floats(
	section_array_input_data: dict,
) -> None:
	input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(
		section_array_input_data
	)
	section = SectionArray(
		input_df, sagging_parameter=2_000, sagging_temperature=15
	)

	assert_frame_equal(input_df, section._data, check_dtype=False, rtol=1e-07)


def test_create_section_array__only_ints() -> None:
	input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(
		{
			"name": ["support 1", "2", "three", "support 4"],
			"suspension": [False, True, True, False],
			"conductor_attachment_altitude": [2, 5, 0, 0],
			"crossarm_length": [10, 12, 10, 10],
			"line_angle": [0, 360, 90, -90],
			"insulator_length": [0, 4, 3, 0],
			"span_length": [1, 500, 500, np.nan],
		}
	)
	section = SectionArray(
		input_df, sagging_parameter=2_000, sagging_temperature=15
	)

	assert_frame_equal(input_df, section._data, check_dtype=False, rtol=1e-07)


def test_create_section_array__span_length_for_last_support(
	section_array_input_data: dict,
) -> None:
	section_array_input_data["span_length"][-1] = 300
	input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(
		section_array_input_data
	)

	with pytest.raises(pa.errors.SchemaErrors):
		SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


@pytest.mark.parametrize(
	"column",
	[
		"name",
		"suspension",
		"conductor_attachment_altitude",
		"crossarm_length",
		"line_angle",
		"insulator_length",
		"span_length",
	],
)
def test_create_section_array__missing_column(
	section_array_input_data: dict, column: str
) -> None:
	del section_array_input_data[column]
	input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(
		section_array_input_data
	)

	with pytest.raises(pa.errors.SchemaErrors):
		SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__extra_column(
	section_array_input_data: dict,
) -> None:
	section_array_input_data["extra column"] = [0] * 4

	input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(
		section_array_input_data
	)

	section = SectionArray(
		input_df, sagging_parameter=2_000, sagging_temperature=15
	)

	assert "extra column" not in section._data.columns


@pytest.mark.parametrize(
	"column,value",
	[
		("name", [1, 2, 3, 4]),
		("suspension", [1, 2, 3, 4]),
		("conductor_attachment_altitude", ["1,2"] * 4),
		("crossarm_length", ["1,2"] * 4),
		("line_angle", ["1,2"] * 4),
		("insulator_length", ["1,2"] * 4),
		("span_length", ["1,2"] * 4),
	],
)
def test_create_section_array__wrong_type(
	section_array_input_data: dict, column: str, value
) -> None:
	section_array_input_data[column] = value
	input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(
		section_array_input_data
	)

	with pytest.raises(pa.errors.SchemaErrors):
		SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__insulator_length_for_tension_support(
	section_array_input_data: dict,
) -> None:
	section_array_input_data["suspension"] = [False, False, True, False]
	section_array_input_data["insulator_length"] = [0.5, 0.5, 0.5, 0.5]
	input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(
		section_array_input_data
	)

	with pytest.raises(pa.errors.SchemaErrors):
		SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_compute_elevation_difference(section_array_input_data: dict) -> None:
	# I modify data to have more meaning here and understand results
	data = {
		"name": ["support 1", "2", "three", "support 4"],
		"suspension": [False, True, True, False],
		"conductor_attachment_altitude": [50.0, 40.0, 20.0, 10.0],
		"crossarm_length": [
			5.0,
		]
		* 4,
		"line_angle": [
			0,
		]
		* 4,
		"insulator_length": [0, 4.0, 3.0, 0],
		"span_length": [50, 100, 500, np.nan],
	}

	df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(data)

	section_array = SectionArray(
		df, sagging_parameter=2_000, sagging_temperature=15
	)

	elevation_difference = section_array.compute_elevation_difference()

	assert_allclose(
		elevation_difference, np.array([-14.0, -19.0, -7.0, np.nan])
	)


def test_section_array__data(section_array_input_data: dict) -> None:
	df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(
		section_array_input_data
	)
	section_array = SectionArray(
		data=df, sagging_parameter=2_000, sagging_temperature=15
	)
	inner_data = section_array._data.copy()

	exported_data = section_array.data

	expected_data = pd.DataFrame(
		{
			"name": ["support 1", "2", "three", "support 4"],
			"suspension": [False, True, True, False],
			"conductor_attachment_altitude": [2.2, 5, -0.12, 0],
			"crossarm_length": [10, 12.1, 10, 10.1],
			"line_angle": [0, 360, 90.1, -90.2],
			"insulator_length": [0, 4, 3.2, 0],
			"span_length": [1, 500.2, 500.05, np.nan],
			"elevation_difference": [-1.2, -4.32, 3.32, np.nan],
			"sagging_parameter": [2_000] * 4,
			"sagging_temperature": [15] * 4,
		},
	)

	assert_frame_equal(exported_data, expected_data, rtol=1e-07)
	# section_array inner data shouldn't have been modified
	assert_frame_equal(section_array._data, inner_data)


def test_section_array__data_without_sagging_properties(
	section_array_input_data: dict,
) -> None:
	df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(
		section_array_input_data
	)

	section_array_without_temperature = SectionArray(
		data=df, sagging_parameter=2_000
	)
	with pytest.raises(AttributeError):
		section_array_without_temperature.data

	section_array_without_parameter = SectionArray(
		data=df, sagging_temperature=15
	)
	with pytest.raises(AttributeError):
		section_array_without_parameter.data


def test_section_array__data_alone(section_array_input_data: dict) -> None:
	df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(
		section_array_input_data
	)
	section_array = SectionArray(data=df)

	exported_data = section_array.data_alone

	expected_data = pd.DataFrame(
		{
			"name": ["support 1", "2", "three", "support 4"],
			"suspension": [False, True, True, False],
			"conductor_attachment_altitude": [2.2, 5, -0.12, 0],
			"crossarm_length": [10, 12.1, 10, 10.1],
			"line_angle": [0, 360, 90.1, -90.2],
			"insulator_length": [0, 4, 3.2, 0],
			"span_length": [1, 500.2, 500.05, np.nan],
			"elevation_difference": [-1.2, -4.32, 3.32, np.nan],
		},
	)

	assert_frame_equal(exported_data, expected_data, rtol=1e-07)


def test_create_cable_array__with_floats(
	cable_array_input_data: dict,
) -> None:
	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)
	cable = CableArray(input_df)

	assert_frame_equal(input_df, cable._data, check_dtype=False, rtol=1e-07)


@pytest.mark.parametrize(
	"column",
	[
		"section",
		"diameter",
		"linear_weight",
		"young_modulus",
		"dilatation_coefficient",
		"temperature_reference",
	],
)
def test_create_cable_array__missing_column(
	cable_array_input_data: dict, column: str
) -> None:
	del cable_array_input_data[column]
	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)

	with pytest.raises(pa.errors.SchemaErrors):
		CableArray(input_df)


@pytest.mark.parametrize(
	"column,value",
	[
		("section", ["1,2"] * 4),
		("diameter", ["1,2"] * 4),
		("linear_weight", ["1,2"] * 4),
		("young_modulus", ["1,2"] * 4),
		("dilatation_coefficient", ["1,2"] * 4),
		("temperature_reference", ["1,2"] * 4),
	],
)
def test_create_cable_array__wrong_type(
	cable_array_input_data: dict, column: str, value
) -> None:
	cable_array_input_data[column] = value
	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		cable_array_input_data
	)

	with pytest.raises(pa.errors.SchemaErrors):
		CableArray(input_df)


def test_cable_array__get_poly_coefs() -> None:
	input_df: pdt.DataFrame[CableArrayInput] = pdt.DataFrame(
		{
		"section": [345.5, 345.5, 345.5, 345.5],
		"diameter": [22.4, 22.4, 22.4, 22.4],
		"linear_weight": [9.6, 9.6, 9.6, 9.6],
		"young_modulus": [59, 59, 59, 59],
		"dilatation_coefficient": [23, 23, 23, 23],
		"temperature_reference": [15, 15, 15, 15],
		"a0" : [3,3,3,3],
		"a1" : [10,10,10,10],
		"a2" : [25,25,25,25],
		"a3" : [400,400,400,400],
		"a4" : [1000,1000,1000,1000],
	})
	
	cable_array = CableArray(input_df)
	polynom = cable_array.polynom
	expected_coefs = np.array([3, 10, 25, 400, 1000])
	assert np.array_equal(polynom.coef, expected_coefs)