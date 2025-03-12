# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from mechaphlowers.api.state import StateAccessor


class MockDeformation:
	def L_ref(self, current_temperature):
		return current_temperature


class MockSection:
	def __init__(self, data_shape, deformation=None):
		self.section = self
		self.data = np.zeros(data_shape)
		self.deformation = deformation


@pytest.fixture
def section_dataframe_without_deformation():
	return MockSection((5,), None)


@pytest.fixture
def section_dataframe():
	return MockSection((5,), MockDeformation())


# ---------Tests---------


def test_Deformation_is_not_defined(section_dataframe_without_deformation):
	state_accessor = StateAccessor(section_dataframe_without_deformation)
	with pytest.raises(
		ValueError,
	):
		state_accessor.L_ref(25.0)


def test_L_ref_with_wrong_type(section_dataframe):
	state_accessor = StateAccessor(section_dataframe)
	current_temperature = "25.0"
	with pytest.raises(ValueError):
		state_accessor.L_ref(current_temperature)


def test_L_ref_with_float_or_int(section_dataframe):
	state_accessor = StateAccessor(section_dataframe)
	current_temperature = 25.0
	result = state_accessor.L_ref(current_temperature)
	current_temperature = 25
	result = state_accessor.L_ref(current_temperature)
	expected = np.full(5, current_temperature)
	np.testing.assert_array_equal(result, expected)


def test_L_ref_with_correct_array(section_dataframe):
	state_accessor = StateAccessor(section_dataframe)
	current_temperature = np.array([25.0, 26.0, 27.0, 28.0, 29.0])
	result = state_accessor.L_ref(current_temperature)
	np.testing.assert_array_equal(result, current_temperature)


def test_L_ref_with_incorrect_array_length(section_dataframe):
	state_accessor = StateAccessor(section_dataframe)
	current_temperature = np.array([25.0, 26.0])
	with pytest.raises(
		ValueError,
		match="Current temperature should have the same length as the section",
	):
		state_accessor.L_ref(current_temperature)
