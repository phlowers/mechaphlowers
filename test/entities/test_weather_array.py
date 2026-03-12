# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pandera as pa
import pytest
from pandas.testing import assert_frame_equal

from mechaphlowers.entities.arrays import (
    WeatherArray,
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
