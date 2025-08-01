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

from mechaphlowers.api.frames import MultipleSection, SectionDataFrame
from mechaphlowers.core.models.cable.span import (
    CatenarySpan,
)
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
    WeatherArray,
)


# To avoid mypy returning error
class CableLoadsInputDict(TypedDict, total=False):
    diameter: np.ndarray
    linear_weight: np.ndarray
    ice_thickness: np.ndarray
    wind_pressure: np.ndarray


def test_section_frame_initialization(
    default_section_array_three_spans,
) -> None:
    frame = SectionDataFrame(default_section_array_three_spans)
    assert frame.section_array == default_section_array_three_spans
    assert isinstance(frame._span_model, type(CatenarySpan))


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


def test_select_spans__passing_input(
    default_section_array_three_spans,
) -> None:
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


def test_SectionDataFrame__copy(default_section_array_three_spans) -> None:
    frame = SectionDataFrame(default_section_array_three_spans)
    copy(frame)
    assert isinstance(frame, SectionDataFrame)


def test_SectionDataFrame__state(
    default_cable_array, default_section_array_three_spans
):
    frame = SectionDataFrame(default_section_array_three_spans)
    frame.add_cable(default_cable_array)
    assert np.array_equal(
        frame.state.L_ref(),
        frame.deformation.L_ref(),  # type: ignore[union-attr]
        equal_nan=True,
    )


# test add_cable method
def test_SectionDataFrame__add_cable(
    default_cable_array, default_section_array_three_spans
):
    frame = SectionDataFrame(default_section_array_three_spans)
    with pytest.raises(TypeError):
        # wrong input type
        frame.add_cable(1)  # type: ignore[arg-type]
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
                "ice_thickness": [1, 2.1, 0.0, np.nan],
                "wind_pressure": [1840.12, 0.0, 12.0, np.nan],
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
                "ice_thickness": [1, 2.1, 0.0, np.nan],
                "wind_pressure": [1840.12, 0.0, 12.0, np.nan],
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
        frame._add_array(default_cable_array._data, pd.DataFrame)  # type: ignore[arg-type]
    # Testez les exceptions
    with pytest.raises(TypeError):
        frame._add_array("not_an_array", CableArray)  # type: ignore[arg-type]


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
        == frame.cable.data.shape[1] + frame.section_array.data.shape[1]  # type: ignore[union-attr]
    )
    assert frame.data.dilatation_coefficient.iloc[-1] == 23e-6
    assert frame.data.a1.iloc[-1] == 59e9
    assert frame.data.b1.iloc[-1] == 0


def test_SectionDataFrame__add_weather_update_span(
    default_cable_array, default_section_array_three_spans
):
    frame = SectionDataFrame(default_section_array_three_spans)
    weather_dict = {
        "ice_thickness": np.array([1, 2.1, 0.0, np.nan]),
        "wind_pressure": np.array([1840.12, 0.0, 12.0, np.nan]),
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
    np.testing.assert_equal(
        frame.span.load_coefficient, cable_loads.load_coefficient
    )
    np.testing.assert_equal(frame.deformation.cable_length, frame.span.L())  # type: ignore[union-attr]


def test_frame__sagtension_use(section_dataframe_with_cable_weather) -> None:
    """Test that the sag tension is calculated correctly."""
    wa = WeatherArray(
        pd.DataFrame(
            {
                "ice_thickness": [1, 2.1, 0.0, np.nan],
                "wind_pressure": [240.12, 0.0, 12.0, np.nan],
            }
        )
    )
    section_dataframe_with_cable_weather.state.change(100, wa)

    assert section_dataframe_with_cable_weather.state.p_after_change.shape == (
        4,
    )
    assert section_dataframe_with_cable_weather.state.L_after_change.shape == (
        4,
    )
    assert (
        section_dataframe_with_cable_weather.state.T_h_after_change.shape
        == (4,)
    )


def test_frame__update_paramater_wind(
    section_dataframe_with_cable_weather,
) -> None:
    """Test that the sag tension is calculated correctly."""
    wa = WeatherArray(
        pd.DataFrame(
            {
                "ice_thickness": [1, 2.1, 0.0, np.nan],
                "wind_pressure": [240.12, 0.0, 12.0, np.nan],
            }
        )
    )
    x = np.linspace(-200, 200, 20)
    section_dataframe_with_cable_weather.state.change(100, wa)
    z_span_after_state_change = section_dataframe_with_cable_weather.span.z(x)

    expected_span = CatenarySpan(
        **section_dataframe_with_cable_weather.data_container.__dict__
    )
    expected_span.sagging_parameter = (
        section_dataframe_with_cable_weather.state.p_after_change
    )
    expected_z = expected_span.z(x)

    np.testing.assert_allclose(
        section_dataframe_with_cable_weather.span.sagging_parameter,
        expected_span.sagging_parameter,
    )
    np.testing.assert_allclose(z_span_after_state_change, expected_z)


def test_multiple_frames(
    default_cable_array,
    generic_weather_array_three_spans,
    factory_neutral_weather_array,
):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["support 1", "2", "three", "support 4"] * 2),
                "suspension": np.array([False, True, True, False] * 2),
                "conductor_attachment_altitude": np.array(
                    [2.2, 5, -0.12, 0] * 2
                ),
                "crossarm_length": np.array([10, 12.1, 10, 10.1] * 2),
                "line_angle": np.array([0, 360, 90.1, -90.2] * 2),
                "insulator_length": np.array([0, 4, 3.2, 0] * 2),
                "span_length": np.array([400, 500.2, 500.0, np.nan] * 2),
                "section_number": np.array([1] * 4 + [2] * 4),
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    cable_array_1 = CableArray(
        pd.DataFrame(
            {
                "section": [500.0],
                "diameter": [30.0],
                "linear_weight": [13.0],
                "young_modulus": [59],
                "dilatation_coefficient": [23],
                "temperature_reference": [15],
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
        )
    )

    weather_array_empty = factory_neutral_weather_array(4)

    multiple_section = MultipleSection(section_array)
    multiple_section.add_cable_all(default_cable_array)
    multiple_section.add_weather_all(weather_array_empty)
    multiple_section.add_cable(1, cable_array_1)
    multiple_section.add_weather(2, generic_weather_array_three_spans)
