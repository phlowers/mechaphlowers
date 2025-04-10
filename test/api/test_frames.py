# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from copy import copy
from typing import Type, TypedDict

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.api.frames import SectionDataFrame
from mechaphlowers.config import options as cfg
from mechaphlowers.core.models.cable.span import (
	CatenarySpan,
)
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import (
	CableArray,
	WeatherArray,
)


# To avoid mypy returning error
class CableLoadsInputDict(TypedDict, total=False):
	diameter: np.ndarray
	linear_weight: np.ndarray
	ice_thickness: np.ndarray
	wind_pressure: np.ndarray


def test_section_frame_initialization(default_section_array_three_spans):
	frame = SectionDataFrame(default_section_array_three_spans)
	assert frame.section_array == default_section_array_three_spans
	assert isinstance(frame._span_model, type(CatenarySpan))


def test_section_frame_get_coord(default_section_array_three_spans):
	frame = SectionDataFrame(default_section_array_three_spans)
	coords = frame.get_coord()
	assert coords.shape == (
		(len(default_section_array_three_spans.data) - 1)
		* cfg.graphics.resolution,
		3,
	)
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
def test_select_spans__wrong_input(
	error: Type[Exception], case, default_section_array_three_spans
):
	frame = SectionDataFrame(default_section_array_three_spans)

	with pytest.raises(error):
		frame.select(case)


def test_select_spans__passing_input(default_section_array_three_spans):
	frame = SectionDataFrame(default_section_array_three_spans)
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


def test_SectionDataFrame__copy(default_section_array_three_spans):
	frame = SectionDataFrame(default_section_array_three_spans)
	copy(frame)
	assert isinstance(frame, SectionDataFrame)


def test_SectionDataFrame__state(
	default_cable_array, default_section_array_three_spans
):
	frame = SectionDataFrame(default_section_array_three_spans)
	frame.add_cable(default_cable_array)
	assert np.array_equal(
		frame.state.L_ref(), frame.deformation.L_ref(), equal_nan=True
	)


# test add_cable method
def test_SectionDataFrame__add_cable(
	default_cable_array, default_section_array_three_spans
):
	frame = SectionDataFrame(default_section_array_three_spans)
	with pytest.raises(TypeError):
		# wrong input type
		frame.add_cable(1)
	with pytest.raises(NotImplementedError):
		wrong_length_array = CableArray(
			default_cable_array._data.loc[
				np.repeat(default_cable_array._data.index, 3)
			].reset_index(drop=True)
		)
		frame.add_cable(wrong_length_array)
	frame.add_cable(default_cable_array)


def test_SectionDataFrame__add_weather(
	default_cable_array, default_section_array_three_spans
):
	frame = SectionDataFrame(default_section_array_three_spans)
	weather = WeatherArray(
		pd.DataFrame(
			{
				"ice_thickness": [1, 2.1, 0.0, 5.4],
				"wind_pressure": [1840.12, 0.0, 12.0, 53.0],
			}
		)
	)
	# cable has to be added before weather
	with pytest.raises(ValueError):
		frame.add_weather(weather)
	# wrong input length
	with pytest.raises(ValueError):
		weather_copy = copy(weather)
		weather_copy._data = weather_copy.data.iloc[:-1]
		frame.add_weather(weather_copy)
	frame.add_cable(cable=default_cable_array)
	frame.add_weather(weather=weather)


def test_SectionDataFrame__add_array(
	default_cable_array, default_section_array_three_spans
):
	frame = SectionDataFrame(default_section_array_three_spans)
	weather_array = WeatherArray(
		pd.DataFrame(
			{
				"ice_thickness": [1, 2.1, 0.0, 5.4],
				"wind_pressure": [1840.12, 0.0, 12.0, 53.0],
			}
		)
	)
	# Testez l'ajout de CableArray
	frame._add_array(default_cable_array, CableArray)
	assert frame.cable == default_cable_array

	# Testez l'ajout de WeatherArray
	frame._add_array(weather_array, WeatherArray)
	assert frame.weather == weather_array

	# Wrong object type
	with pytest.raises(TypeError):
		frame._add_array(default_cable_array._data, pd.DataFrame)
	# Testez les exceptions
	with pytest.raises(TypeError):
		frame._add_array("not_an_array", CableArray)


def test_select_spans__after_added_arrays(
	default_section_array_three_spans,
	default_cable_array,
	factory_neutral_weather_array,
):
	frame = SectionDataFrame(default_section_array_three_spans)
	frame.add_cable(default_cable_array)
	frame.add_weather(factory_neutral_weather_array(4))
	frame_selected = frame.select(["support 1", "three"])
	assert len(frame_selected.data) == 3
	assert (
		frame_selected.data.elevation_difference.take([1]).item()
		== frame.data.elevation_difference.take([1]).item()
	)


def test_SectionDataFrame__data(
	default_cable_array, default_section_array_three_spans
):
	frame = SectionDataFrame(default_section_array_three_spans)
	assert frame.data.equals(frame.section_array.data)

	frame.add_cable(default_cable_array)
	assert not frame.data.equals(frame.section_array.data)
	assert (
		frame.data.shape[1]
		== frame.cable.data.shape[1] + frame.section_array.data.shape[1]
	)
	assert frame.data.dilatation_coefficient.iloc[-1] == 23e-6
	assert frame.data.a1.iloc[-1] == 59e9
	assert frame.data.b1.iloc[-1] == 0


def test_SectionDataFrame__add_weather_update_span(
	default_cable_array, default_section_array_three_spans
):
	frame = SectionDataFrame(default_section_array_three_spans)
	weather_dict = {
		"ice_thickness": np.array([1, 2.1, 0.0, 5.4]),
		"wind_pressure": np.array([1840.12, 0.0, 12.0, 53.0]),
	}
	weather = WeatherArray(pd.DataFrame(weather_dict))
	cable_loads_input = {
		"diameter": default_cable_array.data.diameter.to_numpy(),
		"linear_weight": default_cable_array.data.linear_weight.to_numpy(),
	}
	# Converts into SI units because CableArray automatically converts into SI units but not CableLoads
	cable_loads_input.update(weather_dict)
	cable_loads_input["ice_thickness"] *= 1e-2
	cable_loads = CableLoads(**cable_loads_input)
	frame.add_cable(cable=default_cable_array)
	frame.add_weather(weather=weather)
	assert (frame.span.load_coefficient == cable_loads.load_coefficient).all()
	assert (frame.deformation.cable_length == frame.span.L())[0:-1].all()
