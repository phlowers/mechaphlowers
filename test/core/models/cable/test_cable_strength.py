# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Tests for the AdditiveLayerRts / ITensileStrength model in cable_strength.py."""

from copy import copy

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.config import options
from mechaphlowers.core.models.cable.cable_strength import AdditiveLayerRts
from mechaphlowers.data.catalog.catalog import sample_cable_catalog
from mechaphlowers.entities.arrays import CableArray
from mechaphlowers.entities.errors import RtsDataNotAvailable

# ---------------------------------------------------------------------------
# Integration tests  (rely on the sample catalog and the balance engine fixture
# from conftest.py)
# ---------------------------------------------------------------------------


@pytest.mark.integration_test
def test_rrts() -> None:
    """With no cut strands, RRTS equals rts_cable from the catalog."""
    cable: CableArray = sample_cable_catalog.get_as_object(["ASTER600"])
    assert cable.rrts == 200000


@pytest.mark.integration_test
def test_coverage_rrts() -> None:
    """rts_coverage is close to 0.976 for ASTER600 (sum of strand RTS / cable RTS)."""
    cable: CableArray = sample_cable_catalog.get_as_object(["ASTER600"])
    cable_rrts = cable.rts_coverage()
    assert abs(cable_rrts - 0.976) < 0.01


@pytest.mark.integration_test
def test_cut_strands(balance_engine_base_test) -> None:
    """cut_strands reduces RRTS and changes utilization rate accordingly."""
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
def test_high_safety_rrts() -> None:
    """high_safety multiplies the safety coefficient by 1.5."""
    cable: CableArray = copy(sample_cable_catalog.get_as_object(["ASTER600"]))

    # create the condition to use the default safety coefficient by setting it to None in the data
    cable._data["safety_coefficient"] = (
        None  # Set a known safety coefficient in the data
    )
    cable._tensile_strength = AdditiveLayerRts(
        cable._data
    )  # Recreate the strength model to pick up the change
    options.data.safety_coefficient_default = 2.5
    options.data.safety_security_factor = 10

    assert cable.safety_coefficient == 2.5

    cable.high_safety = True
    assert cable.safety_coefficient == 25

    # set back to original values to avoid side effects on other tests
    options.data.safety_coefficient_default = 1.5
    options.data.safety_security_factor = 1.5


@pytest.mark.integration_test
def test_utilization_rate(balance_engine_base_test) -> None:
    """utilization_rate returns expected percentages for ASTER600."""
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
# Fixtures shared by unit tests
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
def rts_strength_fixture(
    cable_array_with_rts_input_data: dict,
) -> AdditiveLayerRts:
    """AdditiveLayerRts built directly from a unit-converted CableArray snapshot."""
    cable_data = CableArray(pd.DataFrame(cable_array_with_rts_input_data)).data
    return AdditiveLayerRts(cable_data)


# ---------------------------------------------------------------------------
# Unit tests — targeting AdditiveLayerRts directly
# ---------------------------------------------------------------------------


@pytest.mark.unit_test
def test_rrts_no_damage(rts_strength_fixture: AdditiveLayerRts) -> None:
    """With no cut strands, RRTS equals rts_cable."""
    rts_strength_fixture.cut_strands = np.zeros(8, dtype=int)
    assert rts_strength_fixture.rrts == pytest.approx(float(_RTS_CABLE_N))


@pytest.mark.unit_test
def test_rrts_with_damage(rts_strength_fixture: AdditiveLayerRts) -> None:
    """RRTS is reduced by cut strands in layers 1 and 2."""
    rts_strength_fixture.cut_strands = np.array([2, 1, 0, 0, 0, 0, 0, 0])
    expected = float(_RTS_CABLE_N - 2 * _RTS_L1_N - 1 * _RTS_L2_N)
    assert rts_strength_fixture.rrts == pytest.approx(expected)


@pytest.mark.unit_test
def test_rrts_default_no_cut_strands(
    rts_strength_fixture: AdditiveLayerRts,
) -> None:
    """rrts defaults to rts_cable when no cut strands are set."""
    assert rts_strength_fixture.rrts == pytest.approx(float(_RTS_CABLE_N))


@pytest.mark.unit_test
def test_cut_strands_getter(rts_strength_fixture: AdditiveLayerRts) -> None:
    """cut_strands getter returns the padded array previously set."""
    rts_strength_fixture.cut_strands = np.array([2, 1])
    expected = np.array([2, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(rts_strength_fixture.cut_strands, expected)


@pytest.mark.unit_test
def test_cut_strands_too_many_elements(
    rts_strength_fixture: AdditiveLayerRts,
) -> None:
    """cut_strands setter with more than 8 elements raises ValueError."""
    with pytest.raises(ValueError, match="8"):
        rts_strength_fixture.cut_strands = np.zeros(9, dtype=int)


@pytest.mark.unit_test
def test_rrts_raises_if_rts_layer_missing(
    cable_array_with_rts_input_data: dict,
) -> None:
    """rrts raises RtsDataNotAvailable when a cut layer has NaN RTS."""
    data = cable_array_with_rts_input_data.copy()
    data["rts_layer_3"] = [None]
    cable_data = CableArray(pd.DataFrame(data)).data
    strength = AdditiveLayerRts(cable_data)
    strength.cut_strands = np.array([0, 0, 1, 0, 0, 0, 0, 0])
    with pytest.raises(RtsDataNotAvailable, match="rts_layer_3"):
        _ = strength.rrts


@pytest.mark.unit_test
def test_safety_coefficient_default(
    rts_strength_fixture: AdditiveLayerRts,
) -> None:
    """safety_coefficient returns the catalog value when high_safety is False."""
    assert rts_strength_fixture.safety_coefficient == pytest.approx(
        _SAFETY_COEF
    )


@pytest.mark.unit_test
def test_safety_coefficient_high_safety(
    rts_strength_fixture: AdditiveLayerRts,
) -> None:
    """safety_coefficient returns catalog value × 1.5 when high_safety is True."""
    rts_strength_fixture.high_safety = True
    assert rts_strength_fixture.safety_coefficient == pytest.approx(
        _SAFETY_COEF * 1.5
    )


@pytest.mark.unit_test
def test_high_safety_getter_default(
    rts_strength_fixture: AdditiveLayerRts,
) -> None:
    """high_safety getter returns False by default."""
    assert rts_strength_fixture.high_safety is False


@pytest.mark.unit_test
def test_high_safety_getter_after_set(
    rts_strength_fixture: AdditiveLayerRts,
) -> None:
    """high_safety getter reflects the value after being set."""
    rts_strength_fixture.high_safety = True
    assert rts_strength_fixture.high_safety is True


@pytest.mark.unit_test
def test_high_safety_setter_non_bool_raises(
    rts_strength_fixture: AdditiveLayerRts,
) -> None:
    """high_safety setter raises TypeError when a non-bool is passed."""
    with pytest.raises(TypeError, match="boolean"):
        rts_strength_fixture.high_safety = 1  # type: ignore[assignment]


@pytest.mark.unit_test
def test_safety_coefficient_missing_column(
    cable_array_input_data: dict,
) -> None:
    """safety_coefficient falls back to default when column is absent."""
    cable_data = CableArray(pd.DataFrame(cable_array_input_data)).data
    strength = AdditiveLayerRts(cable_data)
    assert strength.safety_coefficient == pytest.approx(
        options.data.safety_coefficient_default
    )


@pytest.mark.unit_test
def test_safety_coefficient_nan_value(
    cable_array_input_data: dict,
) -> None:
    """safety_coefficient falls back to default when value is NaN."""
    data = cable_array_input_data.copy()
    data["safety_coefficient"] = [None]
    cable_data = CableArray(pd.DataFrame(data)).data
    strength = AdditiveLayerRts(cable_data)
    assert strength.safety_coefficient == pytest.approx(
        options.data.safety_coefficient_default
    )


@pytest.mark.unit_test
def test_rts_coverage_no_rts_cable(
    cable_array_with_rts_input_data: dict,
) -> None:
    """rts_coverage raises RtsDataNotAvailable when rts_cable is NaN."""
    data = cable_array_with_rts_input_data.copy()
    data["rts_cable"] = [None]
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
    cable_data = CableArray(pd.DataFrame(data)).data
    strength = AdditiveLayerRts(cable_data)
    with pytest.raises(RtsDataNotAvailable, match="rts_cable"):
        strength.rts_coverage()


@pytest.mark.unit_test
def test_cut_strands_negative_raises(
    rts_strength_fixture: AdditiveLayerRts,
) -> None:
    """cut_strands setter raises ValueError for negative values."""
    with pytest.raises(ValueError, match="non-negative"):
        rts_strength_fixture.cut_strands = np.array([-1, 0, 0, 0, 0, 0, 0, 0])


@pytest.mark.unit_test
def test_cut_strands_exceeds_max_raises(
    cable_array_with_rts_input_data: dict,
) -> None:
    """cut_strands setter raises ValueError when cut > MAX_ALLOWED_CUT_STRANDS (when > 0)."""
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
    cable_data = CableArray(pd.DataFrame(data)).data
    strength = AdditiveLayerRts(cable_data)

    # Set an explicit maximum for layer 1 = 5; passing 6 should raise
    strength.MAX_ALLOWED_CUT_STRANDS = np.array(
        [5, 0, 0, 0, 0, 0, 0, 0], dtype=int
    )
    with pytest.raises(ValueError, match="exceeds allowed maximum"):
        strength.cut_strands = np.array([6, 0, 0, 0, 0, 0, 0, 0])


@pytest.mark.unit_test
def test_cut_all_strands_should_not_raise(
    cable_array_with_rts_input_data: dict,
) -> None:
    """cut_strands setter should allow cutting all strands (cut = nb_strands)."""
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
    cable_data = CableArray(pd.DataFrame(data)).data
    strength = AdditiveLayerRts(cable_data)

    # Set maximum to the number of strands; this should not raise
    strength.MAX_ALLOWED_CUT_STRANDS = np.array(
        [10, 8, 6, 4, 0, 0, 0, 0], dtype=int
    )
