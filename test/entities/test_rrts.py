# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Test module for RRTS entity with real usecases"""

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.config import options
from mechaphlowers.data.catalog.catalog import sample_cable_catalog
from mechaphlowers.entities.arrays import CableArray
from mechaphlowers.entities.errors import RtsDataNotAvailable


@pytest.mark.integration_test
def test_rrts() -> None:
    """Test RRTS entity with real usecases"""
    cable: CableArray = sample_cable_catalog.get_as_object(["ASTER600"])

    assert cable.rrts == 200000


@pytest.mark.integration_test
def test_coverage_rrts() -> None:
    """Test coverage of RRTS entity with real usecases"""
    cable: CableArray = sample_cable_catalog.get_as_object(["ASTER600"])

    cable_rrts = cable.rts_coverage()
    assert abs(cable_rrts - 1.02) < 0.01


@pytest.mark.integration_test
def test_cut_strands(balance_engine_base_test):
    """Test cut_strands method of RRTS entity with real usecases"""
    cable: CableArray = sample_cable_catalog.get_as_object(["ASTER600"])

    nb_strands = cable.nb_strand_per_layer

    assert nb_strands.shape == (8,)

    cable.cut_strands = np.array([5, 3, 2, 0, 0, 0, 0, 0])
    assert abs(cable.rrts - 168000) < 1.0

    balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(0, 0, 15)
    tension_max, _ = balance_engine_base_test.span_model.tensions_sup_inf()
    utilization_rate = cable.utilization_rate(tension_max)

    expected_utilization_rate = np.array([32.3, 32.3, 31.8, np.nan])

    np.testing.assert_array_almost_equal(
        utilization_rate, expected_utilization_rate, decimal=1
    )


@pytest.mark.integration_test
def test_high_safety_rrts():
    """Test high_safety_rrts method of RRTS entity with real usecases"""
    cable: CableArray = sample_cable_catalog.get_as_object(["ASTER600"])
    options.data.safety_coefficient = 1.5
    options.data.safety_security_factor = 1.5

    assert cable.safety_coefficient == 1.5

    cable.high_safety = True
    assert cable.safety_coefficient == 2.25


@pytest.mark.integration_test
def test_utilization_rate(balance_engine_base_test) -> None:
    """Test utilization_rate method of RRTS entity with real usecases"""
    cable = sample_cable_catalog.get_as_object(["ASTER600"])
    balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(0, 0, 15)
    tension_max, _ = balance_engine_base_test.span_model.tensions_sup_inf()
    utilization_rate = cable.utilization_rate(tension_max)

    expected_utilization_rate = np.array([27.2, 27.2, 26.7, np.nan])

    np.testing.assert_array_almost_equal(
        utilization_rate, expected_utilization_rate, decimal=1
    )

    # Test with safety coefficient
    cable.high_safety = True
    utilization_rate = cable.utilization_rate(tension_max)
    np.testing.assert_array_almost_equal(
        utilization_rate, np.array([40.7, 40.7, 40.0, np.nan]), decimal=1
    )


# ---------------------------------------------------------------------------
# RRTS / unit tests
# ---------------------------------------------------------------------------

_RTS_CABLE_N = 18000  # N (arbitrary test value)
_RTS_L1_N = 2500  # N (arbitrary test value)
_RTS_L2_N = 5000  # N (arbitrary test value)
_SAFETY_COEF = 1.5


@pytest.fixture
def cable_array_input_data() -> dict:
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


@pytest.fixture
def cable_array_with_rts_input_data(cable_array_input_data: dict) -> dict:
    data = cable_array_input_data.copy()
    data.update(
        {
            "rts_cable": [_RTS_CABLE_N],
            "rts_layer_1": [_RTS_L1_N],
            "rts_layer_2": [_RTS_L2_N],
            "rts_layer_3": [7500],
            "rts_layer_4": [3000],
            "rts_layer_5": [0],
            "rts_layer_6": [0],
            "rts_layer_7": [0],
            "rts_layer_8": [0],
            "safety_coefficient": [_SAFETY_COEF],
        }
    )
    return data


@pytest.fixture
def cable_array_with_rts(cable_array_with_rts_input_data: dict) -> CableArray:
    return CableArray(pd.DataFrame(cable_array_with_rts_input_data))


@pytest.mark.unit_test
def test_rrts_no_damage(cable_array_with_rts: CableArray) -> None:
    """With no cut strands, RRTS equals rts_cable converted to N."""
    cable_array_with_rts.cut_strands = np.zeros(8, dtype=int)
    expected_rrts_N = float(_RTS_CABLE_N)
    assert cable_array_with_rts.rrts == pytest.approx(expected_rrts_N)


@pytest.mark.unit_test
def test_rrts_with_damage(cable_array_with_rts: CableArray) -> None:
    """RRTS is reduced by cut strands in layers 1 and 2."""
    # 2 strands cut in layer 1, 1 strand cut in layer 2
    cable_array_with_rts.cut_strands = np.array([2, 1, 0, 0, 0, 0, 0, 0])
    expected_rrts_N = float(_RTS_CABLE_N - 2 * _RTS_L1_N - 1 * _RTS_L2_N)
    assert cable_array_with_rts.rrts == pytest.approx(expected_rrts_N)


@pytest.mark.unit_test
def test_rrts_default_no_cut_strands(cable_array_with_rts: CableArray) -> None:
    """rrts defaults to rts_cable when no cut strands are set."""
    expected_rrts_N = float(_RTS_CABLE_N)
    assert cable_array_with_rts.rrts == pytest.approx(expected_rrts_N)


@pytest.mark.unit_test
def test_cut_strands_getter(cable_array_with_rts: CableArray) -> None:
    """cut_strands getter returns the padded array previously set."""
    cable_array_with_rts.cut_strands = np.array([2, 1])
    expected = np.array([2, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(cable_array_with_rts.cut_strands, expected)


@pytest.mark.unit_test
def test_cut_strands_too_many_elements(
    cable_array_with_rts: CableArray,
) -> None:
    """cut_strands setter with more than 8 elements raises ValueError."""
    with pytest.raises(ValueError, match="8"):
        cable_array_with_rts.cut_strands = np.zeros(9, dtype=int)


@pytest.mark.unit_test
def test_rrts_raises_if_rts_layer_missing(
    cable_array_with_rts_input_data: dict,
) -> None:
    """rrts raises ValueError when a cut layer has NaN RTS in catalog data."""
    # Set rts_layer_3 to None (will become NaN after schema coercion)
    data = cable_array_with_rts_input_data.copy()
    data["rts_layer_3"] = [None]
    ca = CableArray(pd.DataFrame(data))
    # Cut 1 strand in layer 3 (index 2 → rts_layer_3)
    ca.cut_strands = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="rts_layer_3"):
        _ = ca.rrts


@pytest.mark.unit_test
def test_safety_coefficient_default(cable_array_with_rts: CableArray) -> None:
    """safety_coefficient returns the catalog value when high_safety is False."""
    assert cable_array_with_rts.safety_coefficient == pytest.approx(
        _SAFETY_COEF
    )


@pytest.mark.unit_test
def test_safety_coefficient_high_safety(
    cable_array_with_rts: CableArray,
) -> None:
    """safety_coefficient returns catalog value × 1.5 when high_safety is True."""
    cable_array_with_rts.high_safety = True
    assert cable_array_with_rts.safety_coefficient == pytest.approx(
        _SAFETY_COEF * 1.5
    )


# ---------------------------------------------------------------------------
# New tests covering lines 442, 447, 461, 565, 572, 602, 617, 621
# ---------------------------------------------------------------------------


@pytest.mark.unit_test
def test_high_safety_getter_default(cable_array_with_rts: CableArray) -> None:
    """high_safety getter (line 442) returns False by default."""
    assert cable_array_with_rts.high_safety is False


@pytest.mark.unit_test
def test_high_safety_getter_after_set(
    cable_array_with_rts: CableArray,
) -> None:
    """high_safety getter (line 442) reflects the value after being set."""
    cable_array_with_rts.high_safety = True
    assert cable_array_with_rts.high_safety is True


@pytest.mark.unit_test
def test_high_safety_setter_non_bool_raises(
    cable_array_with_rts: CableArray,
) -> None:
    """high_safety setter (line 447) raises TypeError when a non-bool is passed."""
    with pytest.raises(TypeError, match="boolean"):
        cable_array_with_rts.high_safety = 1  # type: ignore[assignment]


@pytest.mark.unit_test
def test_safety_coefficient_missing_column(
    cable_array_input_data: dict,
) -> None:
    """safety_coefficient (line 461) falls back to default when column is absent."""
    ca = CableArray(pd.DataFrame(cable_array_input_data))
    assert ca.safety_coefficient == pytest.approx(
        options.data.safety_coefficient_default
    )


@pytest.mark.unit_test
def test_safety_coefficient_nan_value(
    cable_array_input_data: dict,
) -> None:
    """safety_coefficient (line 461) falls back to default when value is NaN."""
    data = cable_array_input_data.copy()
    data["safety_coefficient"] = [None]
    ca = CableArray(pd.DataFrame(data))
    assert ca.safety_coefficient == pytest.approx(
        options.data.safety_coefficient_default
    )


@pytest.mark.unit_test
def test_rts_coverage_zero_denominator(
    cable_array_with_rts_input_data: dict,
) -> None:
    """rts_coverage (line 565) raises ValueError when denominator is zero."""
    # nb_strand_* columns absent → all zeros → denominator = 0
    ca = CableArray(pd.DataFrame(cable_array_with_rts_input_data))
    with pytest.raises(ValueError, match="denominator is zero"):
        ca.rts_coverage()


@pytest.mark.unit_test
def test_rts_coverage_no_rts_cable(
    cable_array_with_rts_input_data: dict,
) -> None:
    """rts_coverage (line 572) raises RtsDataNotAvailable when rts_cable is NaN."""
    data = cable_array_with_rts_input_data.copy()
    data["rts_cable"] = [None]
    # Provide nb_strand_* so denominator is non-zero
    data.update(
        {
            "nb_strand_layer_1": [10],
            "nb_strand_layer_2": [8],
            "nb_strand_layer_3": [6],
            "nb_strand_layer_4": [4],
            "nb_strand_layer_5": [0],
            "nb_strand_layer_6": [0],
            "nb_strand_layer_7": [0],
            "nb_strand_layer_8": [0],
        }
    )
    ca = CableArray(pd.DataFrame(data))
    with pytest.raises(RtsDataNotAvailable, match="rts_cable"):
        ca.rts_coverage()


@pytest.mark.unit_test
def test_cut_strands_negative_raises(cable_array_with_rts: CableArray) -> None:
    """cut_strands setter (line 602) raises ValueError for negative values."""
    with pytest.raises(ValueError, match="non-negative"):
        cable_array_with_rts.cut_strands = np.array([-1, 0, 0, 0, 0, 0, 0, 0])


@pytest.mark.unit_test
def test_cut_strands_exceeds_max_raises(
    cable_array_with_rts_input_data: dict,
) -> None:
    """cut_strands setter (lines 617/621) raises ValueError when cut > max allowed."""
    data = cable_array_with_rts_input_data.copy()
    data.update(
        {
            "nb_strand_layer_1": [10],
            "nb_strand_layer_2": [8],
            "nb_strand_layer_3": [6],
            "nb_strand_layer_4": [4],
            "nb_strand_layer_5": [0],
            "nb_strand_layer_6": [0],
            "nb_strand_layer_7": [0],
            "nb_strand_layer_8": [0],
        }
    )
    ca = CableArray(pd.DataFrame(data))
    # max allowed for layer 1 = int(10/2) = 5; passing 6 should raise
    with pytest.raises(ValueError, match="exceeds allowed maximum"):
        ca.cut_strands = np.array([6, 0, 0, 0, 0, 0, 0, 0])
