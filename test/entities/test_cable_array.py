# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import pandera as pa
import pytest
from pandas.testing import assert_frame_equal

from mechaphlowers.entities.arrays import (
    CableArray,
)


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
            "electric_resistance_20": [5.54e-5],
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

    cable_array = CableArray(input_df)

    assert "extra column" not in cable_array._data.columns


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
            "electric_resistance_20": [5.54e-5],
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
            "electric_resistance_20": [5.54e-5],
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
