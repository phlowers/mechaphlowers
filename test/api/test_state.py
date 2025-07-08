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


class MockWeatherArray:
    def __init__(self, wind_speed, wind_direction, ice_thickness):
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.ice_thickness = ice_thickness


@pytest.fixture
def weather_array():
    return MockWeatherArray(
        wind_speed=10, wind_direction=180, ice_thickness=0.5
    )


@pytest.fixture
def section_dataframe_without_deformation():
    return MockSection((5,), None)


@pytest.fixture
def section_dataframe():
    return MockSection((5,), MockDeformation())


# ---------Tests---------


def test_Deformation_is_not_defined(
    section_dataframe_without_deformation,
) -> None:
    state_accessor = StateAccessor(section_dataframe_without_deformation)
    with pytest.raises(ValueError):
        state_accessor.L_ref()


def test_L_ref_value(section_dataframe) -> None:
    state_accessor = StateAccessor(section_dataframe)
    result = state_accessor.L_ref()
    expected = np.array([1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_change_without_deformation(
    section_dataframe_without_deformation, weather_array
):
    state_accessor = StateAccessor(section_dataframe_without_deformation)
    with pytest.raises(
        ValueError,
    ):
        state_accessor.change(
            np.array([25.0, 26.0, 27.0, 28.0, 29.0]), weather_array
        )


def test_change_with_deformation(
    section_dataframe_with_cable_weather, generic_weather_array_three_spans
):
    state_accessor = StateAccessor(section_dataframe_with_cable_weather)
    current_temperature = np.array(
        [
            25.0,
            26.0,
            27.0,
            28.0,
        ]
    )
    state_accessor.change(
        current_temperature, generic_weather_array_three_spans
    )
    assert state_accessor.sag_tension is not None
    assert isinstance(state_accessor.p_after_change, np.ndarray)
    assert isinstance(state_accessor.L_after_change, np.ndarray)


# TODO: add tests for change() testing current_temperature type
