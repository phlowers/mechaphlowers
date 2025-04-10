# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from mechaphlowers.api.state import StateAccessor


class MockDeformation:
	def L_ref(self):
		return np.array([1, 2, 3])


class MockSection:
	def __init__(self, data_shape, deformation=None):
		self.section_array = self
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
	with pytest.raises(ValueError):
		state_accessor.L_ref()


def test_L_ref_value(section_dataframe):
	state_accessor = StateAccessor(section_dataframe)
	result = state_accessor.L_ref()
	expected = np.array([1, 2, 3])
	np.testing.assert_array_equal(result, expected)
