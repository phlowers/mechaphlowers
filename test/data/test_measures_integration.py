# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd
import pytest

from mechaphlowers import PapotoParameterMeasure
from mechaphlowers.data.measures import param_calibration
from mechaphlowers.entities.arrays import CableArray, SectionArray


def test_param_calibr(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": [0, 10, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [100, 50, 50, 100],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )

    param_0 = param_calibration(
        2000, 60, section_array, cable_array_AM600, span_index=0
    )
    np.testing.assert_allclose(param_0, 2543.57334, atol=1e-1)

    param_1 = param_calibration(
        2000, 60, section_array, cable_array_AM600, span_index=1
    )
    np.testing.assert_allclose(param_1, 2573.74948, atol=1e-1)

    param_2 = param_calibration(
        2000, 60, section_array, cable_array_AM600, span_index=2
    )
    np.testing.assert_allclose(param_2, 2578.5492, atol=1e-1)


def test_param_calibr_deg_no_anchor(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": [0, 10, 0, 0],
                "insulator_length": [0.01, 3, 3, 0.01],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [0, 50, 50, 0],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    param_0 = param_calibration(
        2000, 60, section_array, cable_array_AM600, span_index=0
    )
    np.testing.assert_allclose(param_0, 2548.2168, atol=1e-1)

    param_1 = param_calibration(
        2000, 60, section_array, cable_array_AM600, span_index=1
    )
    np.testing.assert_allclose(param_1, 2578.4119, atol=1e-1)

    param_2 = param_calibration(
        2000, 60, section_array, cable_array_AM600, span_index=2
    )
    np.testing.assert_allclose(param_2, 2583.5351, atol=1e-1)


def test_param_calibr_deg_simple_example(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [50, 50, 50, 50],
                "crossarm_length": [0, 0, 0, 0],
                "line_angle": [0, 0, 0, 0],
                "insulator_length": [0.01, 3, 3, 0.01],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [0, 0, 0, 0],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )

    param_0 = param_calibration(
        2000, 60, section_array, cable_array_AM600, span_index=0
    )
    np.testing.assert_allclose(param_0, 2566.2089, atol=1e-1)

    param_1 = param_calibration(
        2000, 60, section_array, cable_array_AM600, span_index=1
    )
    np.testing.assert_allclose(param_1, 2596.6778, atol=1e-1)

    param_2 = param_calibration(
        2000, 60, section_array, cable_array_AM600, span_index=2
    )
    np.testing.assert_allclose(param_2, 2601.3522, atol=1e-1)


def test_param_calibr_deg_elevation_diff(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 0, 0, 0],
                "line_angle": [0, 0, 0, 0],
                "insulator_length": [0.01, 3, 3, 0.01],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [100, 50, 50, 100],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )

    param_0 = param_calibration(
        2000, 60, section_array, cable_array_AM600, span_index=0
    )
    np.testing.assert_allclose(param_0, 2565.5791, atol=1e-1)

    param_1 = param_calibration(
        2000, 60, section_array, cable_array_AM600, span_index=1
    )
    np.testing.assert_allclose(param_1, 2597.9831, atol=1e-1)

    param_2 = param_calibration(
        2000, 60, section_array, cable_array_AM600, span_index=2
    )
    np.testing.assert_allclose(param_2, 2603.1956, atol=1e-1)


@pytest.mark.integration
def test_uncertainty_integration():
    """Integration test for uncertainty() with realistic inputs."""

    PAPOTO_INPUTS = dict(
        a=500.0,
        HL=309.3920,
        VL=97.5154,
        HR=458.0377,
        VR=74.0039,
        H1=316.0746,
        V1=96.2049,
        H2=333.1511,
        V2=89.8211,
        H3=395.4711,
        V3=71.2413,
    )

    papoto = PapotoParameterMeasure()
    papoto(**PAPOTO_INPUTS)

    # checks for memory
    np.testing.assert_allclose(papoto.parameter[0], 1957.1, atol=1e-1)
    np.testing.assert_allclose(papoto.validity[0], 0.004, atol=1e-3)
    papoto.angle_unit = "rad"
    result = papoto.uncertainty(draw_number=1000, angle_error=0.01, seed=42)

    assert isinstance(result, dict)

    EXPECTED_UNCERTAINTY_RESULT = {
        'mean_parameter_valid_values': np.float64(1957.7),
        'std_parameter_valid_values': np.float64(2.90),
        'min_parameter_valid_values': np.float64(1949.44),
        'max_parameter_valid_values': np.float64(1965.82),
        'parameter_by_span_length': np.float64(3.92),
        'number_non_valid_values': 377,
        'mean_non_valid_values': np.float64(1955.96),
        'std_non_valid_values': np.float64(2.73),
        'min_all_values': np.float64(1947.68),
        'max_all_values': np.float64(1965.82),
    }

    for key, expected_value in EXPECTED_UNCERTAINTY_RESULT.items():
        assert key in result, f"Missing key '{key}' in uncertainty result"
        actual_value = result[key]
        if isinstance(expected_value, (int, float, np.number)):
            np.testing.assert_allclose(actual_value, expected_value, atol=1e-1)
        else:
            assert (
                actual_value == expected_value
            ), f"Value mismatch for key '{key}'"
