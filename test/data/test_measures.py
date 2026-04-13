# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pytest

from mechaphlowers.data.measures import (
    PapotoParameterMeasure,
    param_calibration,
)
from mechaphlowers.entities.arrays import CableArray, SectionArray

PAPOTO_INPUTS = dict(
    a=498.565922913587,
    HL=0.0,
    VL=97.4327311161033,
    HR=162.614599621714,
    VR=88.6907631859419,
    H1=5.1134354937127,
    V1=98.4518011880176,
    H2=19.6314054626454,
    V2=97.6289296721015,
    H3=97.1475339907774,
    V3=87.9335010245142,
)


def test_papoto_floats():
    a = 498.565922913587
    HL = 0.0
    VL = 97.4327311161033
    HR = 162.614599621714
    VR = 88.6907631859419
    H1 = 5.1134354937127
    V1 = 98.4518011880176
    H2 = 19.6314054626454
    V2 = 97.6289296721015
    H3 = 97.1475339907774
    V3 = 87.9335010245142

    papoto = PapotoParameterMeasure()
    papoto(
        a=a,
        HL=HL,
        VL=VL,
        HR=HR,
        VR=VR,
        H1=H1,
        V1=V1,
        H2=H2,
        V2=V2,
        H3=H3,
        V3=V3,
    )
    np.testing.assert_allclose(
        papoto.parameter, np.array([2000, np.nan]), atol=1.0
    )
    np.testing.assert_allclose(
        papoto.parameter_1_2, np.array([1999.78, np.nan]), atol=0.1
    )
    np.testing.assert_allclose(
        papoto.validity, np.array([8.85880213e-05, np.nan]), atol=1e-5
    )
    np.testing.assert_allclose(
        papoto.check_validity(), np.array([True, False]), atol=0.1
    )


def test_papoto_parameter_measure():
    a = np.array([498.565922913587, np.nan])
    HL = np.array([0.0, np.nan])
    VL = np.array([97.4327311161033, np.nan])
    HR = np.array([162.614599621714, np.nan])
    VR = np.array([88.6907631859419, np.nan])
    H1 = np.array([5.1134354937127, np.nan])
    V1 = np.array([98.4518011880176, np.nan])
    H2 = np.array([19.6314054626454, np.nan])
    V2 = np.array([97.6289296721015, np.nan])
    H3 = np.array([97.1475339907774, np.nan])
    V3 = np.array([87.9335010245142, np.nan])

    papoto = PapotoParameterMeasure()
    papoto(
        a=a,
        HL=HL,
        VL=VL,
        HR=HR,
        VR=VR,
        H1=H1,
        V1=V1,
        H2=H2,
        V2=V2,
        H3=H3,
        V3=V3,
    )

    np.testing.assert_allclose(
        papoto.parameter, np.array([2000, np.nan]), atol=1.0
    )
    np.testing.assert_allclose(
        papoto.parameter_1_2, np.array([1999.78, np.nan]), atol=0.1
    )
    np.testing.assert_allclose(
        papoto.validity, np.array([8.85880213e-05, np.nan]), atol=1e-5
    )
    np.testing.assert_allclose(
        papoto.check_validity(), np.array([True, False]), atol=0.1
    )


def test_select_points_in_dict():
    # Prepare mock data
    data = {
        "a": np.array([1]),
        "HL": np.array([2]),
        "VL": np.array([3]),
        "HR": np.array([4]),
        "VR": np.array([5]),
        "H1": np.array([10]),
        "V1": np.array([11]),
        "H2": np.array([20]),
        "V2": np.array([21]),
        "H3": np.array([30]),
        "V3": np.array([31]),
    }

    # Select points 1 and 3
    result = PapotoParameterMeasure.select_points_in_dict(1, 3, data)

    # Check that only H1/V1 and H3/V3 are present, and H2/V2 are replaced
    assert result["H1"].tolist() == [10]
    assert result["V1"].tolist() == [11]
    assert result["H2"].tolist() == [30]
    assert result["V2"].tolist() == [31]
    # Other keys should be present and unchanged
    assert result["a"].tolist() == [1]
    assert result["HL"].tolist() == [2]
    assert result["VL"].tolist() == [3]
    assert result["HR"].tolist() == [4]
    assert result["VR"].tolist() == [5]
    # H3/V3 should not be present as keys
    assert "H3" not in result
    assert "V3" not in result

    # Select points 2 and 1
    result2 = PapotoParameterMeasure.select_points_in_dict(2, 1, data)
    assert result2["H1"].tolist() == [20]
    assert result2["V1"].tolist() == [21]
    assert result2["H2"].tolist() == [10]
    assert result2["V2"].tolist() == [11]


def test_parameter_15_deg(
    section_array_complete: SectionArray, cable_array_AM600: CableArray
):
    # checks that no error is raised
    param_calibration(
        2000, 60, section_array_complete, cable_array_AM600, span_index=0
    )

    param_calibration(
        2000, 60, section_array_complete, cable_array_AM600, span_index=1
    )

    param_calibration(
        2000, 60, section_array_complete, cable_array_AM600, span_index=2
    )


EXPECTED_UNCERTAINTY_KEYS = {
    'mean_parameter_valid_values',
    'std_parameter_valid_values',
    'min_parameter_valid_values',
    'max_parameter_valid_values',
    'parameter_by_span_length',
    'number_non_valid_values',
    'mean_non_valid_values',
    'std_non_valid_values',
    'min_all_values',
    'max_all_values',
}


def test_uncertainty_raises_before_measure_method():
    """uncertainty() must raise RuntimeError if measure_method() not called first."""
    papoto = PapotoParameterMeasure()
    with pytest.raises(RuntimeError):
        papoto.uncertainty()


def test_uncertainty_returns_expected_keys():
    """uncertainty() must return a dict with all expected keys."""
    papoto = PapotoParameterMeasure()
    papoto(**PAPOTO_INPUTS)
    result = papoto.uncertainty(draw_number=100)
    assert isinstance(result, dict)
    assert set(result.keys()) == EXPECTED_UNCERTAINTY_KEYS


def test_uncertainty_number_non_valid_is_int():
    """number_non_valid_values must be an integer."""
    papoto = PapotoParameterMeasure()
    papoto(**PAPOTO_INPUTS)
    result = papoto.uncertainty(draw_number=100)
    assert isinstance(result['number_non_valid_values'], int)


def test_uncertainty_valid_values_consistent():
    """mean/std/min/max over valid draws must be internally consistent."""
    papoto = PapotoParameterMeasure()
    papoto(**PAPOTO_INPUTS)

    result = papoto.uncertainty(draw_number=500, angle_error=0.01, seed=42)

    assert (
        result['min_parameter_valid_values']
        <= result['mean_parameter_valid_values']
    )
    assert (
        result['mean_parameter_valid_values']
        <= result['max_parameter_valid_values']
    )
    assert result['std_parameter_valid_values'] >= 0
    assert result['min_all_values'] <= result['max_all_values']


def test_uncertainty_parameter_by_span_length():
    """parameter_by_span_length must equal mean_valid / a."""
    papoto = PapotoParameterMeasure()
    papoto(**PAPOTO_INPUTS)
    result = papoto.uncertainty(draw_number=200, seed=42)
    expected = result['mean_parameter_valid_values'] / PAPOTO_INPUTS['a']
    np.testing.assert_allclose(result['parameter_by_span_length'], expected)


def test_uncertainty_non_valid_nan_when_all_valid():
    """When all draws are valid, non-valid stats should be NaN."""
    papoto = PapotoParameterMeasure()
    papoto(**PAPOTO_INPUTS)

    # Use very small angle_error so all draws should be valid
    result = papoto.uncertainty(draw_number=200, angle_error=1e-6)

    if result['number_non_valid_values'] == 0:
        assert np.isnan(result['mean_non_valid_values'])
        assert np.isnan(result['std_non_valid_values'])