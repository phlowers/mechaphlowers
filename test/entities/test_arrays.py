# Copyright (c) 2024, RTE (http://www.rte-france.com)
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

from mechaphlowers.entities.arrays import SectionArray, SectionArrayInput


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
    return SectionArray(data=df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__with_floats(section_array_input_data: dict) -> None:
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)
    section = SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)

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
    section = SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)

    assert_frame_equal(input_df, section._data, check_dtype=False, rtol=1e-07)


def test_create_section_array__span_length_for_last_support(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["span_length"][-1] = 300
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__missing_name(section_array_input_data: dict) -> None:
    del section_array_input_data["name"]
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__missing_suspension(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["suspension"]
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__missing_conductor_attachment_altitude(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["conductor_attachment_altitude"]
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__missing_crossarm_length(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["crossarm_length"]
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__missing_line_angle(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["line_angle"]
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__missing_insulator_length(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["insulator_length"]
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__missing_span_length(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["span_length"]
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__extra_column(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["extra column"] = [0] * 4

    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    section = SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)

    assert "extra column" not in section._data.columns


def test_create_section_array__wrong_name_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["name"] = [1, 2, 3, 4]
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__wrong_suspension_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["suspension"] = [1, 2, 3, 4]
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__wrong_conductor_attachment_altitude_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["conductor_attachment_altitude"] = ["1,2"] * 4
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__wrong_crossarm_length_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["crossarm_length"] = ["1,2"] * 4
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__wrong_line_angle_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["line_angle"] = ["1,2"] * 4
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__wrong_insulator_length_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["insulator_length"] = ["1,2"] * 4
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__wrong_span_length_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["span_length"] = ["1,2"] * 4
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__insulator_length_for_tension_support(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["suspension"] = [False, False, True, False]
    section_array_input_data["insulator_length"] = [0.5, 0.5, 0.5, 0.5]
    input_df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_compute_elevation_difference(section_array_input_data: dict) -> None:
    df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    section_array = SectionArray(df, sagging_parameter=2_000, sagging_temperature=15)

    elevation_difference = section_array.compute_elevation_difference()

    assert_allclose(elevation_difference, np.array([-2.8, 5.12, -0.12, np.nan]))


def test_section_array__data(section_array_input_data: dict) -> None:
    df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)
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
            "elevation_difference": [-2.8, 5.12, -0.12, np.nan],
            "sagging_parameter": [2_000] * 4,
            "sagging_temperature": [15] * 4,
        },
    )

    assert_frame_equal(exported_data, expected_data, rtol=1e-07)
    # section_array inner data shouldn't have been modified
    assert_frame_equal(section_array._data, inner_data)


def test_section_array__without_sagging_properties(
    section_array_input_data: dict,
) -> None:
    df: pdt.DataFrame[SectionArrayInput] = pdt.DataFrame(section_array_input_data)

    section_array_without_temperature = SectionArray(data=df, sagging_parameter=2_000)
    with pytest.raises(Exception):
        section_array_without_temperature.data

    section_array_without_parameter = SectionArray(data=df, sagging_temperature=15)
    with pytest.raises(Exception):
        section_array_without_parameter.data
