# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from copy import copy
from typing import Callable, Type, TypedDict

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.api.frames import RESOLUTION, SectionDataFrame
from mechaphlowers.core.models.cable.span import (
	CatenarySpan,
)
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import (
	CableArray,
	SectionArray,
	WeatherArray,
)


class CableLoadsInputDict(TypedDict, total=False):
	diameter: np.ndarray
	linear_weight: np.ndarray
	ice_thickness: np.ndarray
	wind_pressure: np.ndarray


data = {
	"name": ["support 1", "2", "three", "support 4"],
	"suspension": [False, True, True, False],
	"conductor_attachment_altitude": [2.2, 5, -0.12, 0],
	"crossarm_length": [10, 12.1, 10, 10.1],
	"line_angle": [0, 360, 90.1, -90.2],
	"insulator_length": [0, 4, 3.2, 0],
	"span_length": [1, 500.2, 500.0, np.nan],
}

section = SectionArray(data=pd.DataFrame(data))
section.sagging_parameter = 2000
section.sagging_temperature = 15


def test_section_frame_initialization():
	frame = SectionDataFrame(section)
	assert frame.section_array == section
	assert isinstance(frame._span_model, type(CatenarySpan))


def test_section_frame_get_coord():
	frame = SectionDataFrame(section)
	coords = frame.get_coord()
	assert coords.shape == ((len(section.data) - 1) * RESOLUTION, 3)
	assert isinstance(coords, np.ndarray)


@pytest.mark.parametrize(
	"error,case",
	[
		(ValueError, ["support 1", "2", "three"]),
		(ValueError, ["support 1"]),
		(ValueError, ["support 1", "support 1"]),
		(ValueError, ["support 1", "name_not_existing"]),
		(ValueError, ["three", "support 1"]),
		(TypeError, "support 1"),
		(TypeError, ["support 1", 2]),
	],
)
def test_select_spans__wrong_input(error: Type[Exception], case):
	frame = SectionDataFrame(section)

	with pytest.raises(error):
		frame.select(case)


# TODO: Add test on data property


def test_select_spans__passing_input():
	frame = SectionDataFrame(section)
	frame_selected = frame.select(["support 1", "three"])
	assert len(frame_selected.data) == 3
	assert (
		frame_selected.data.elevation_difference.take([1]).item()
		== frame.data.elevation_difference.take([1]).item()
	)

	frame_selected = frame.select(["2", "support 4"])
	assert len(frame_selected.data) == 3
	assert (
		frame_selected.data.elevation_difference.take([1]).item()
		== frame.data.elevation_difference.take([2]).item()
	)


def test_SectionDataFrame__copy():
	frame = SectionDataFrame(section)
	copy(frame)
	assert isinstance(frame, SectionDataFrame)


def test_SectionDataFrame__state():
	frame = SectionDataFrame(section)
	cable_array = CableArray(
		pd.DataFrame(
			{
				"section": [
					345.5,
				]
				* 4,
				"diameter": [
					22.4,
				]
				* 4,
				"linear_weight": [
					9.6,
				]
				* 4,
				"young_modulus": [
					59,
				]
				* 4,
				"dilatation_coefficient": [
					23,
				]
				* 4,
				"temperature_reference": [
					15,
				]
				* 4,
				"a0": [0] * 4,
				"a1": [59] * 4,
				"a2": [0] * 4,
				"a3": [0] * 4,
				"a4": [0] * 4,
				"b0": [0] * 4,
				"b1": [0] * 4,
				"b2": [0] * 4,
				"b3": [0] * 4,
				"b4": [0] * 4,
			}
		)
	)

	frame.add_cable(cable_array)
	assert np.array_equal(
		frame.state.L_ref(12), frame.deformation.L_ref(12), equal_nan=True
	)


# test add_cable method
def test_SectionDataFrame__add_cable():
	frame = SectionDataFrame(section)
	cable_array = CableArray(
		pd.DataFrame(
			{
				"section": [
					345.5,
				]
				* 4,
				"diameter": [
					22.4,
				]
				* 4,
				"linear_weight": [
					9.6,
				]
				* 4,
				"young_modulus": [
					59,
				]
				* 4,
				"dilatation_coefficient": [
					23,
				]
				* 4,
				"temperature_reference": [
					15,
				]
				* 4,
				"a0": [0] * 4,
				"a1": [59] * 4,
				"a2": [0] * 4,
				"a3": [0] * 4,
				"a4": [0] * 4,
				"b0": [0] * 4,
				"b1": [0] * 4,
				"b2": [0] * 4,
				"b3": [0] * 4,
				"b4": [0] * 4,
			}
		)
	)

	with pytest.raises(TypeError):
		# wrong input type
		frame.add_cable(1)
	with pytest.raises(ValueError):
		# wrong input length
		cable_copy = copy(cable_array)
		cable_copy._data = cable_copy.data.iloc[:-1]
		frame.add_cable(cable_copy)
	frame.add_cable(cable_array)


def test_SectionDataFrame__add_weather():
	frame = SectionDataFrame(section)
	weather = WeatherArray(
		pd.DataFrame(
			{
				"ice_thickness": [1, 2.1, 0.0, 5.4],
				"wind_pressure": [1840.12, 0.0, 12.0, 53.0],
			}
		)
	)

	cable_array = CableArray(
		pd.DataFrame(
			{
				"section": [
					345.5,
				]
				* 4,
				"diameter": [
					22.4,
				]
				* 4,
				"linear_weight": [
					9.6,
				]
				* 4,
				"young_modulus": [
					59,
				]
				* 4,
				"dilatation_coefficient": [
					23,
				]
				* 4,
				"temperature_reference": [
					15,
				]
				* 4,
				"a0": [0] * 4,
				"a1": [59] * 4,
				"a2": [0] * 4,
				"a3": [0] * 4,
				"a4": [0] * 4,
				"b0": [0] * 4,
				"b1": [0] * 4,
				"b2": [0] * 4,
				"b3": [0] * 4,
				"b4": [0] * 4,
			}
		)
	)
	with pytest.raises(ValueError):
		# cable has to be added before weather
		frame.add_weather(weather)
	frame.add_cable(cable=cable_array)
	frame.add_weather(weather=weather)


def test_SectionDataFrame__add_array():
	frame = SectionDataFrame(section)
	cable_df = pd.DataFrame(
		{
			"section": [
				345.5,
			]
			* 4,
			"diameter": [
				22.4,
			]
			* 4,
			"linear_weight": [
				9.6,
			]
			* 4,
			"young_modulus": [
				59,
			]
			* 4,
			"dilatation_coefficient": [
				23,
			]
			* 4,
			"temperature_reference": [
				15,
			]
			* 4,
			"a0": [0] * 4,
			"a1": [59] * 4,
			"a2": [0] * 4,
			"a3": [0] * 4,
			"a4": [0] * 4,
			"b0": [0] * 4,
			"b1": [0] * 4,
			"b2": [0] * 4,
			"b3": [0] * 4,
			"b4": [0] * 4,
		}
	)
	cable_array = CableArray(cable_df)
	weather_array = WeatherArray(
		pd.DataFrame(
			{
				"ice_thickness": [1, 2.1, 0.0, 5.4],
				"wind_pressure": [1840.12, 0.0, 12.0, 53.0],
			}
		)
	)
	# Testez l'ajout de CableArray
	frame._add_array(cable_array, CableArray)
	assert frame.cable == cable_array

	# Testez l'ajout de WeatherArray
	frame._add_array(weather_array, WeatherArray)
	assert frame.weather == weather_array

	# Wrong object type
	with pytest.raises(TypeError):
		frame._add_array(cable_df, pd.DataFrame)
	# Testez les exceptions
	with pytest.raises(TypeError):
		frame._add_array("not_an_array", CableArray)

	with pytest.raises(ValueError):
		wrong_length_array = copy(cable_array)
		wrong_length_array._data = wrong_length_array.data.iloc[:-1]
		frame._add_array(wrong_length_array, CableArray)


def test_SectionDataFrame__data():
	cable_array = CableArray(
		pd.DataFrame(
			{
				"section": [
					345.5,
				]
				* 4,
				"diameter": [
					22.4,
				]
				* 4,
				"linear_weight": [
					9.6,
				]
				* 4,
				"young_modulus": [
					59,
				]
				* 4,
				"dilatation_coefficient": [
					23,
				]
				* 4,
				"temperature_reference": [
					15,
				]
				* 4,
				"a0": [0] * 4,
				"a1": [59] * 4,
				"a2": [0] * 4,
				"a3": [0] * 4,
				"a4": [0] * 4,
				"b0": [0] * 4,
				"b1": [0] * 4,
				"b2": [0] * 4,
				"b3": [0] * 4,
				"b4": [0] * 4,
			}
		)
	)

	frame = SectionDataFrame(section)
	# TODO: fix this test: recreate expected result by changing "name" -> "support_name"
	expected_dict = frame.section_array.data.rename(columns ={"name": "support_name"})
	assert frame.data.equals(expected_dict)

	frame.add_cable(cable_array)
	assert not frame.data.equals(frame.section_array.data)
	assert (
		frame.data.shape[1]
		== frame.cable.data.shape[1]
		+ frame.section_array.data.shape[1]
		+ 2  # both polynomials add 2 more attributes
	)


def test_SectionDataFrame__add_weather_update_span(
	factory_cable_array: Callable[[int], CableArray],
):
	cable_array = factory_cable_array(4)
	frame = SectionDataFrame(section)
	weather_dict: CableLoadsInputDict = {
		"ice_thickness": np.array([1, 2.1, 0.0, 5.4]),
		"wind_pressure": np.array([1840.12, 0.0, 12.0, 53.0]),
	}
	weather = WeatherArray(pd.DataFrame(weather_dict))
	cable_loads_input: CableLoadsInputDict = {
		"diameter": cable_array.data.diameter.to_numpy(),
		"linear_weight": cable_array.data.linear_weight.to_numpy(),
	}
	# Converts into SI units because CableArray automatically converts into SI units but not CableLoads
	cable_loads_input.update(weather_dict)
	cable_loads_input["ice_thickness"] *= 1e-2
	cable_loads = CableLoads(**cable_loads_input)
	frame.add_cable(cable=cable_array)
	frame.add_weather(weather=weather)
	assert (frame.span.load_coefficient == cable_loads.load_coefficient).all()
	assert (frame.deformation.cable_length == frame.span.L())[0:-1].all()
