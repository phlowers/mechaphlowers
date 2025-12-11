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

from mechaphlowers.config import options
from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
    WeatherArray,
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
        "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
    }


@pytest.fixture
def section_array(section_array_input_data: dict[str, list]) -> SectionArray:
    df = pd.DataFrame(section_array_input_data)
    section_array = SectionArray(
        data=df, sagging_parameter=2_000, sagging_temperature=15
    )
    section_array.add_units({"line_angle": "deg"})
    return section_array


@pytest.fixture
def cable_array_input_data() -> dict[str, list]:
    return {
        "section": [600.4],
        "diameter": [31.86],
        "linear_mass": [1.8],
        "young_modulus": [60000],
        "dilatation_coefficient": [23e-6],
        "temperature_reference": [15.0],
        "a0": [0.0],
        "a1": [60000],
        "a2": [0.0],
        "a3": [0.0],
        "a4": [0.0],
        "b0": [0.0],
        "b1": [0.0],
        "b2": [0.0],
        "b3": [0.0],
        "b4": [0.0],
        "diameter_heart": [0.0],
        "section_conductor": [600.4],
        "section_heart": [0.0],
        "solar_absorption": [0.9],
        "emissivity": [0.8],
        "electric_resistance_20": [0.0554],
        "linear_resistance_temperature_coef": [0.0036],
        "is_polynomial": [False],
        "radial_thermal_conductivity": [1.0],
        "has_magnetic_heart": [False],
    }


def test_create_section_array__with_floats(
    section_array_input_data: dict,
) -> None:
    input_df = pd.DataFrame(section_array_input_data)
    section = SectionArray(
        input_df, sagging_parameter=2_000, sagging_temperature=15
    )

    assert_frame_equal(input_df, section._data, check_dtype=False, atol=1e-07)


def test_create_section_array__only_ints() -> None:
    input_df = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2, 5, 0, 0],
            "crossarm_length": [10, 12, 10, 10],
            "line_angle": [0, 360, 90, -90],
            "insulator_length": [0, 4, 3, 0],
            "span_length": [1, 500, 500, np.nan],
            "insulator_mass": np.array([1000.0, 500.0, 500.0, 1000.0]),
        }
    )
    section = SectionArray(
        input_df, sagging_parameter=2_000, sagging_temperature=15
    )

    assert_frame_equal(input_df, section._data, check_dtype=False, atol=1e-07)


def test_create_section_array__span_length_for_last_support(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["span_length"][-1] = 300
    input_df = pd.DataFrame(section_array_input_data)

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
        "insulator_mass",
    ],
)
def test_create_section_array__missing_column(
    section_array_input_data: dict, column: str
) -> None:
    del section_array_input_data[column]
    input_df = pd.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_create_section_array__extra_column(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["extra column"] = [0] * 4

    input_df = pd.DataFrame(section_array_input_data)

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
        ("insulator_mass", ["1,2"] * 4),
    ],
)
def test_create_section_array__wrong_type(
    section_array_input_data: dict, column: str, value
) -> None:
    section_array_input_data[column] = value
    input_df = pd.DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df, sagging_parameter=2_000, sagging_temperature=15)


def test_compute_elevation_difference() -> None:
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
        "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
    }

    df = pd.DataFrame(data)

    section_array = SectionArray(
        df, sagging_parameter=2_000, sagging_temperature=15
    )

    elevation_difference = section_array.compute_elevation_difference()

    assert_allclose(
        elevation_difference, np.array([-10.0, -20.0, -10.0, np.nan])
    )


def test_section_array__data(section_array_input_data: dict) -> None:
    df = pd.DataFrame(section_array_input_data)
    section_array = SectionArray(
        data=df, sagging_parameter=2_000, sagging_temperature=15
    )
    section_array.add_units({"line_angle": "deg"})
    inner_data = section_array._data.copy()

    exported_data = section_array.data

    expected_data = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
            "crossarm_length": [10, 12.1, 10, 10.1],
            "line_angle": [0.0, 6.283185, 1.572542, -1.574287],
            "insulator_length": [0, 4, 3.2, 0],
            "span_length": [1, 500.2, 500.05, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
            "insulator_weight": [9810.0, 4905.0, 4905.0, 9810.0],
            "ground_altitude": [-27.8, -25.0, -30.12, -30.0],
            "elevation_difference": [2.8, -5.12, 0.12, np.nan],
            "sagging_parameter": [2_000.0, 2_000.0, 2_000.0, np.nan],
            "sagging_temperature": [15] * 4,
        },
    )

    assert_frame_equal(exported_data, expected_data, atol=1e-07)
    # section_array inner data shouldn't have been modified
    assert_frame_equal(section_array._data, inner_data)


def test_section_array__data_with_optional() -> None:
    df = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
            "crossarm_length": [10, 12.1, 10, 10.1],
            "line_angle": [0, 360, 90.1, -90.2],
            "insulator_length": [0, 4, 3.2, 0],
            "span_length": [1, 500.2, 500.05, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
            "load_mass": [500, 1000, 500, np.nan],
            "load_position": [0.2, 0.4, 0.6, np.nan],
            "ground_altitude": [0.0, 3.0, -1, 0],
        }
    )
    section_array = SectionArray(
        data=df, sagging_parameter=2_000, sagging_temperature=15
    )
    section_array.add_units({"line_angle": "deg"})

    inner_data = section_array._data.copy()

    exported_data = section_array.data

    expected_data = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
            "crossarm_length": [10, 12.1, 10, 10.1],
            "line_angle": [0.0, 6.283185, 1.572542, -1.574287],
            "insulator_length": [0, 4, 3.2, 0],
            "span_length": [1, 500.2, 500.05, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
            "insulator_weight": [9810.0, 4905.0, 4905.0, 9810.0],
            "elevation_difference": [2.8, -5.12, 0.12, np.nan],
            "ground_altitude": [0.0, 3.0, -1, 0],
            "sagging_parameter": [2_000.0, 2_000.0, 2_000.0, np.nan],
            "sagging_temperature": [15] * 4,
            "load_mass": [500, 1000, 500, np.nan],
            "load_weight": [4905.0, 9810.0, 4905.0, np.nan],
            "load_position": [0.2, 0.4, 0.6, np.nan],
        },
    )

    assert_frame_equal(
        exported_data, expected_data, atol=1e-07, check_like=True
    )
    # section_array inner data shouldn't have been modified
    assert_frame_equal(section_array._data, inner_data, check_like=True)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_section_array__wrong_ground_altitude() -> None:
    df = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
            "crossarm_length": [10, 12.1, 10, 10.1],
            "line_angle": [0, 360, 90.1, -90.2],
            "insulator_length": [0, 4, 3.2, 0],
            "span_length": [1, 500.2, 500.05, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
            "load_mass": [500, 1000, 500, np.nan],
            "load_position": [0.2, 0.4, 0.6, np.nan],
            "ground_altitude": [0.0, 7.0, -1, 5],
        }
    )
    section_array = SectionArray(
        data=df, sagging_parameter=2_000, sagging_temperature=15
    )
    section_array.add_units({"line_angle": "deg"})

    exported_data = section_array.data

    expected_data = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
            "crossarm_length": [10, 12.1, 10, 10.1],
            "line_angle": [0.0, 6.283185, 1.572542, -1.574287],
            "insulator_length": [0, 4, 3.2, 0],
            "span_length": [1, 500.2, 500.05, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
            "insulator_weight": [9810.0, 4905.0, 4905.0, 9810.0],
            "elevation_difference": [2.8, -5.12, 0.12, np.nan],
            "ground_altitude": [0.0, -25.0, -1.0, -30.0],
            "sagging_parameter": [2_000.0, 2_000.0, 2_000.0, np.nan],
            "sagging_temperature": [15] * 4,
            "load_mass": [500, 1000, 500, np.nan],
            "load_weight": [4905.0, 9810.0, 4905.0, np.nan],
            "load_position": [0.2, 0.4, 0.6, np.nan],
        },
    )

    assert_frame_equal(
        exported_data, expected_data, atol=1e-07, check_like=True
    )


def test_section_array__data_without_sagging_properties(
    section_array_input_data: dict,
) -> None:
    df = pd.DataFrame(section_array_input_data)

    section_array_without_temperature = SectionArray(
        data=df, sagging_parameter=2_000
    )
    np.testing.assert_allclose(
        section_array_without_temperature.data.sagging_temperature,
        options.data.sagging_temperature_default
        * np.ones(len(section_array_without_temperature.data)),
    )

    section_array_without_parameter = SectionArray(
        data=df, sagging_temperature=15
    )
    np.testing.assert_allclose(
        section_array_without_parameter.data.sagging_parameter,
        section_array_without_parameter.equivalent_span()
        * 5
        * np.array(
            [1] * (len(section_array_without_temperature.data) - 1) + [np.nan]
        ),
    )


def test_section_array__data_original(section_array_input_data: dict) -> None:
    df = pd.DataFrame(section_array_input_data)
    section_array = SectionArray(data=df)

    exported_data = section_array.data_original

    expected_data = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
            "crossarm_length": [10, 12.1, 10, 10.1],
            "line_angle": [0, 360, 90.1, -90.2],
            "insulator_length": [0, 4, 3.2, 0],
            "span_length": [1, 500.2, 500.05, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
        },
    )

    assert_frame_equal(exported_data, expected_data, atol=1e-07)


def test_create_cable_array__with_floats(
    cable_array_input_data: dict,
) -> None:
    input_df = pd.DataFrame(cable_array_input_data)
    cable = CableArray(input_df)

    assert_frame_equal(cable._data, input_df, check_dtype=False, atol=1e-07)
    expected_result_SI_units = pd.DataFrame(
        {
            "section": [600.4e-6],
            "diameter": [31.86e-3],
            "linear_weight": [17.658],
            "linear_mass": [1.8],
            "young_modulus": [60e9],
            "dilatation_coefficient": [23e-6],
            "temperature_reference": [15.0],
            "a0": [0.0],
            "a1": [60e9],
            "a2": [0.0],
            "a3": [0.0],
            "a4": [0.0],
            "b0": [0.0],
            "b1": [0.0],
            "b2": [0.0],
            "b3": [0.0],
            "b4": [0.0],
            "diameter_heart": [0.0],
            "section_conductor": [600.4e-6],
            "section_heart": [0.0],
            "solar_absorption": [0.9],
            "emissivity": [0.8],
            "electric_resistance_20": [0.0554],
            "linear_resistance_temperature_coef": [0.0036],
            "is_polynomial": [False],
            "radial_thermal_conductivity": [1.0],
            "has_magnetic_heart": [False],
        }
    )
    assert_frame_equal(
        cable.data,
        expected_result_SI_units,
        check_like=True,
        check_dtype=False,
        atol=1e-07,
    )


@pytest.mark.parametrize(
    "column",
    [
        "section",
        "diameter",
        "linear_mass",
        "young_modulus",
        "dilatation_coefficient",
        "temperature_reference",
    ],
)
def test_create_cable_array__missing_column(
    cable_array_input_data: dict, column: str
) -> None:
    del cable_array_input_data[column]
    input_df = pd.DataFrame(cable_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        CableArray(input_df)


@pytest.mark.parametrize(
    "column,value",
    [
        ("section", ["1,2"]),
        ("diameter", ["1,2"]),
        ("linear_mass", ["1,2"]),
        ("young_modulus", ["1,2"]),
        ("dilatation_coefficient", ["1,2"]),
        ("temperature_reference", ["1,2"]),
    ],
)
def test_create_cable_array__wrong_type(
    cable_array_input_data: dict, column: str, value
) -> None:
    cable_array_input_data[column] = value
    input_df = pd.DataFrame(cable_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        CableArray(input_df)


def test_create_cable_array__extra_column(
    cable_array_input_data: dict,
) -> None:
    cable_array_dict_copy = cable_array_input_data.copy()
    cable_array_dict_copy["extra column"] = [0]

    input_df = pd.DataFrame(cable_array_dict_copy)

    section = CableArray(input_df)

    assert "extra column" not in section._data.columns


def test_create_cable_array__units(
    cable_array_input_data: dict,
) -> None:
    input_df = pd.DataFrame(cable_array_input_data)
    cable = CableArray(input_df)

    custom_units = {
        "section": "cm^2",
        "diameter": "cm",
        "young_modulus": "kPa",
    }

    cable.add_units(custom_units)

    expected_result_SI_units = pd.DataFrame(
        {
            "section": [600.4e-4],
            "diameter": [31.86e-2],
            "linear_weight": [17.658],
            "linear_mass": [1.8],
            "young_modulus": [60e6],
            "dilatation_coefficient": [23e-6],
            "temperature_reference": [15.0],
            "a0": [0.0],
            "a1": [60e9],
            "a2": [0.0],
            "a3": [0.0],
            "a4": [0.0],
            "b0": [0.0],
            "b1": [0.0],
            "b2": [0.0],
            "b3": [0.0],
            "b4": [0.0],
            "diameter_heart": [0.0],
            "section_conductor": [600.4e-6],
            "section_heart": [0.0],
            "solar_absorption": [0.9],
            "emissivity": [0.8],
            "electric_resistance_20": [0.0554],
            "linear_resistance_temperature_coef": [0.0036],
            "is_polynomial": [False],
            "radial_thermal_conductivity": [1.0],
            "has_magnetic_heart": [False],
        }
    )
    assert_frame_equal(
        cable.data,
        expected_result_SI_units,
        check_like=True,
        check_dtype=False,
        atol=1e-07,
    )


def test_cable_mecha_thermal_data(cable_array_input_data: dict):
    cable_array = CableArray(pd.DataFrame(cable_array_input_data))

    expected_mecha_data = pd.DataFrame(
        {
            "section": [600.4e-6],
            "diameter": [31.86e-3],
            "linear_weight": [17.658],
            "young_modulus": [60e9],
            "dilatation_coefficient": [23e-6],
            "temperature_reference": [15.0],
            "a0": [0.0],
            "a1": [60e9],
            "a2": [0.0],
            "a3": [0.0],
            "a4": [0.0],
            "b0": [0.0],
            "b1": [0.0],
            "b2": [0.0],
            "b3": [0.0],
            "b4": [0.0],
            "diameter_heart": [0.0],
            "section_conductor": [600.4e-6],
            "section_heart": [0.0],
            "is_polynomial": [False],
        }
    )

    assert_frame_equal(
        cable_array.data_mecha,
        expected_mecha_data,
        check_like=True,
        atol=1e-07,
    )

    expected_thermal_data = pd.DataFrame(
        {
            "diameter": [31.86e-3],
            "linear_weight": [17.658],
            "diameter_heart": [0.0],
            "section_conductor": [600.4e-6],
            "section_heart": [0.0],
            "solar_absorption": [0.9],
            "emissivity": [0.8],
            "electric_resistance_20": [0.0554],
            "linear_resistance_temperature_coef": [0.0036],
            "radial_thermal_conductivity": [1.0],
            "has_magnetic_heart": [False],
        }
    )

    assert_frame_equal(
        cable_array.data_thermal,
        expected_thermal_data,
        check_like=True,
        atol=1e-07,
    )


def test_create_weather_array() -> None:
    weather = WeatherArray(
        pd.DataFrame(
            {
                "ice_thickness": [1, 2.1],
                "wind_pressure": [240.12, 0],
            }
        )
    )

    expected_result_SI_units = pd.DataFrame(
        {
            "ice_thickness": np.array([1e-2, 2.1e-2]),
            "wind_pressure": np.array([240.12, 0]),
        }
    )

    assert_frame_equal(
        weather.data,
        expected_result_SI_units,
        check_like=True,
        check_dtype=False,
        atol=1e-07,
    )


def test_create_weather_array__units() -> None:
    weather = WeatherArray(
        pd.DataFrame(
            {
                "ice_thickness": [1, 2.1],
                "wind_pressure": [240.12, 0],
            }
        )
    )

    weather.add_units({"ice_thickness": "dm"})
    expected_result_SI_units = pd.DataFrame(
        {
            "ice_thickness": np.array([1e-1, 2.1e-1]),
            "wind_pressure": np.array([240.12, 0]),
        }
    )

    assert_frame_equal(
        weather.data,
        expected_result_SI_units,
        check_like=True,
        check_dtype=False,
        atol=1e-07,
    )


def test_create_weather_array__negative_ice() -> None:
    input_data_with_negative_ice = {
        "ice_thickness": [1, -5.0, -0.0001],
        "wind_pressure": [240.12, 0, -240.13],
    }
    input_df = pd.DataFrame(input_data_with_negative_ice)

    with pytest.raises(pa.errors.SchemaErrors):
        WeatherArray(input_df)


def test_equivalent_span(section_array) -> None:
    res = (
        section_array.data.span_length**3
    ).sum() / section_array.data.span_length.sum()

    np.testing.assert_allclose(section_array.equivalent_span() ** 2, res)
