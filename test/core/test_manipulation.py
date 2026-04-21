# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest
import warnings as _warnings
from numpy.testing import assert_allclose

from mechaphlowers.core.manipulation import Manipulation
from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.entities.errors import BalanceEngineWarning


@pytest.fixture
def section_array() -> SectionArray:
    df = pd.DataFrame(
        {
            "name": ["support 1", "2", "three", "support 4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
            "crossarm_length": [10, 12.1, 10, 10.1],
            "line_angle": [0, 360, 90.1, -90.2],
            "insulator_length": [0.01, 4, 3.2, 0.01],
            "span_length": [1, 500.2, 500.05, np.nan],
            "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
        }
    )
    sa = SectionArray(data=df, sagging_parameter=2_000, sagging_temperature=15)
    sa.add_units({"line_angle": "deg"})
    return sa


@pytest.fixture
def manipulation(section_array: SectionArray) -> Manipulation:
    return Manipulation(section_array)


# ── support manipulation ─────────────────────────────────────────────────


def test_support_manipulation_single_support(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_alt = section_array._data["conductor_attachment_altitude"].copy()
    original_arm = section_array._data["crossarm_length"].copy()

    manipulation.support_manipulation({1: {"z": 3.0, "y": -2.0}})
    applied = manipulation.from_section_array(section_array)

    assert_allclose(
        applied.data["conductor_attachment_altitude"].iloc[1],
        original_alt.iloc[1] + 3.0,
    )
    assert_allclose(
        applied.data["crossarm_length"].iloc[1],
        original_arm.iloc[1] - 2.0,
    )
    # Other supports unchanged
    assert_allclose(
        applied.data["conductor_attachment_altitude"].iloc[0],
        original_alt.iloc[0],
    )


def test_support_manipulation_multiple_supports(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_alt = section_array._data["conductor_attachment_altitude"].copy()

    manipulation.support_manipulation(
        {0: {"z": 1.0}, 2: {"z": -0.5, "y": 2.0}}
    )
    applied = manipulation.from_section_array(section_array)

    assert_allclose(
        applied.data["conductor_attachment_altitude"].iloc[0],
        original_alt.iloc[0] + 1.0,
    )
    assert_allclose(
        applied.data["conductor_attachment_altitude"].iloc[2],
        original_alt.iloc[2] - 0.5,
    )


def test_support_manipulation_partial_keys(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_alt = section_array._data["conductor_attachment_altitude"].copy()
    original_arm = section_array._data["crossarm_length"].copy()

    manipulation.support_manipulation({0: {"z": 1.5}})
    applied = manipulation.from_section_array(section_array)

    assert_allclose(
        applied.data["conductor_attachment_altitude"].iloc[0],
        original_alt.iloc[0] + 1.5,
    )
    assert_allclose(
        applied.data["crossarm_length"].iloc[0],
        original_arm.iloc[0],
    )

    manipulation.reset_manipulation()
    manipulation.support_manipulation({0: {"y": -1.0}})
    applied = manipulation.from_section_array(section_array)

    assert_allclose(
        applied.data["conductor_attachment_altitude"].iloc[0],
        original_alt.iloc[0],
    )
    assert_allclose(
        applied.data["crossarm_length"].iloc[0],
        original_arm.iloc[0] - 1.0,
    )


def test_support_manipulation_invalid_index(
    manipulation: Manipulation,
) -> None:
    with pytest.raises(ValueError, match="out of range"):
        manipulation.support_manipulation({10: {"z": 1.0}})


def test_support_manipulation_invalid_keys(
    manipulation: Manipulation,
) -> None:
    with pytest.raises(ValueError, match="Invalid keys"):
        manipulation.support_manipulation({0: {"x": 1.0}})


def test_reset_manipulation(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_alt = section_array.data["conductor_attachment_altitude"].copy()
    original_arm = section_array.data["crossarm_length"].copy()

    manipulation.support_manipulation({0: {"z": 5.0}, 1: {"y": -3.0}})
    manipulation.support_manipulation({2: {"z": 1.0}})
    manipulation.reset_manipulation()
    applied = manipulation.from_section_array(section_array)

    assert_allclose(
        applied.data["conductor_attachment_altitude"].to_numpy(),
        original_alt.to_numpy(),
    )
    assert_allclose(
        applied.data["crossarm_length"].to_numpy(),
        original_arm.to_numpy(),
    )


def test_reset_manipulation_no_prior(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_alt = section_array.data["conductor_attachment_altitude"].copy()

    manipulation.reset_manipulation()  # should not raise
    applied = manipulation.from_section_array(section_array)

    assert_allclose(
        applied.data["conductor_attachment_altitude"].to_numpy(),
        original_alt.to_numpy(),
    )


# ── rope manipulation ────────────────────────────────────────────────────


def test_rope_manipulation_overrides_data(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_insulator_length = section_array._data["insulator_length"].copy()
    original_insulator_mass = section_array._data["insulator_mass"].copy()

    manipulation.rope_manipulation({1: 5.0, 2: 3.0})
    applied = manipulation.from_section_array(section_array)

    # Modified supports
    assert_allclose(applied.data["insulator_length"].iloc[1], 5.0)
    assert_allclose(applied.data["insulator_length"].iloc[2], 3.0)
    assert_allclose(
        applied.data["insulator_mass"].iloc[1], 5.0 * 0.01
    )  # default lineic mass
    assert_allclose(applied.data["insulator_mass"].iloc[2], 3.0 * 0.01)

    # Unlisted support unchanged
    assert_allclose(
        applied.data["insulator_length"].iloc[0],
        original_insulator_length.iloc[0],
    )
    assert_allclose(
        applied.data["insulator_mass"].iloc[0],
        original_insulator_mass.iloc[0],
    )

    # Original _data must be untouched
    assert_allclose(
        section_array._data["insulator_length"].to_numpy(),
        original_insulator_length.to_numpy(),
    )
    assert_allclose(
        section_array._data["insulator_mass"].to_numpy(),
        original_insulator_mass.to_numpy(),
    )


def test_rope_manipulation_custom_lineic_mass(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    manipulation.rope_manipulation({0: 2.0}, rope_lineic_mass=0.5)
    applied = manipulation.from_section_array(section_array)

    assert_allclose(applied.data["insulator_mass"].iloc[0], 2.0 * 0.5)


def test_rope_manipulation_insulator_weight_updated(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    """insulator_weight in .data must reflect the rope mass."""
    manipulation.rope_manipulation({1: 4.0}, rope_lineic_mass=0.1)
    applied = manipulation.from_section_array(section_array)

    expected_weight = 4.0 * 0.1 * 9.81  # approx N
    assert_allclose(
        applied.data["insulator_weight"].iloc[1], expected_weight, rtol=1e-3
    )


def test_rope_manipulation_invalid_index(manipulation: Manipulation) -> None:
    with pytest.raises(ValueError, match="out of range"):
        manipulation.rope_manipulation({99: 3.0})


def test_reset_rope_manipulation(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_length = section_array.data["insulator_length"].copy()
    original_mass = section_array.data["insulator_mass"].copy()

    manipulation.rope_manipulation({1: 5.0, 2: 3.0})
    manipulation.reset_rope_manipulation()
    applied = manipulation.from_section_array(section_array)

    assert_allclose(
        applied.data["insulator_length"].to_numpy(),
        original_length.to_numpy(),
    )
    assert_allclose(
        applied.data["insulator_mass"].to_numpy(), original_mass.to_numpy()
    )


def test_reset_rope_manipulation_no_prior(
    manipulation: Manipulation,
) -> None:
    manipulation.reset_rope_manipulation()  # should not raise


# ── counterweight masking ─────────────────────────────────────────────────


def _make_section_array_with_counterweight() -> SectionArray:
    sa = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 0, 0, 0],
                "line_angle": [0, 0, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [1000, 500, 500, 1000],
                "counterweight_mass": [0, 200, 300, 0],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    sa.add_units({"line_angle": "grad"})
    return sa


def test_counterweight_masked_during_support_manipulation() -> None:
    sa = _make_section_array_with_counterweight()
    original_counterweight = sa.data["counterweight"].copy()
    assert (original_counterweight > 0).any()

    manip = Manipulation(sa)
    manip.support_manipulation({1: {"z": 2.0}})
    applied = manip.from_section_array(sa)
    data = applied.data

    # Only support 1 is affected
    assert_allclose(data["counterweight"].iloc[1], 0.0)
    # Other supports with counterweight are unchanged
    assert_allclose(
        data["counterweight"].iloc[2], original_counterweight.iloc[2]
    )

    manip.reset_manipulation()
    applied = manip.from_section_array(sa)
    assert_allclose(
        applied.data["counterweight"].to_numpy(),
        original_counterweight.to_numpy(),
    )


def test_counterweight_masked_during_rope_manipulation() -> None:
    sa = _make_section_array_with_counterweight()
    original_counterweight = sa.data["counterweight"].copy()
    assert (original_counterweight > 0).any()

    manip = Manipulation(sa)
    manip.rope_manipulation({1: 4.5})
    applied = manip.from_section_array(sa)
    data = applied.data

    # Only support 1 is affected
    assert_allclose(data["counterweight"].iloc[1], 0.0)
    # Other supports with counterweight are unchanged
    assert_allclose(
        data["counterweight"].iloc[2], original_counterweight.iloc[2]
    )

    manip.reset_rope_manipulation()
    applied = manip.from_section_array(sa)
    assert_allclose(
        applied.data["counterweight"].to_numpy(),
        original_counterweight.to_numpy(),
    )


# ── virtual support ──────────────────────────────────────────────────────


def test_add_virtual_support_changes_data_shape(
    section_array: SectionArray,
) -> None:
    manip = Manipulation(section_array)
    assert len(section_array.data) == 4
    manip.add_virtual_support(
        {
            1: {
                "x": 100.0,
                "y": 0.0,
                "z": 55.0,
                "insulator_length": 3.0,
                "insulator_mass": 500.0,
                "hanging_cable_point_from_left_support": 100.0,
            }
        }
    )
    applied = manip.from_section_array(section_array)
    assert len(applied.data) == 5


def test_reset_virtual_support_restores_data_shape(
    section_array: SectionArray,
) -> None:
    manip = Manipulation(section_array)
    manip.add_virtual_support(
        {
            1: {
                "x": 100.0,
                "y": 0.0,
                "z": 55.0,
                "insulator_length": 3.0,
                "insulator_mass": 500.0,
                "hanging_cable_point_from_left_support": 100.0,
            }
        }
    )
    manip.reset_virtual_support()
    applied = manip.from_section_array(section_array)
    assert len(applied.data) == 4


# ── apply preserves original ──────────────────────────────────────────────


def test_apply_does_not_modify_original(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_data = section_array._data.copy()

    manipulation.support_manipulation({0: {"z": 10.0}})
    manipulation.rope_manipulation({1: 6.0})
    _ = manipulation.from_section_array(section_array)

    # Original section array must be untouched
    assert_allclose(
        section_array._data["conductor_attachment_altitude"].to_numpy(),
        original_data["conductor_attachment_altitude"].to_numpy(),
    )
    assert_allclose(
        section_array._data["insulator_length"].to_numpy(),
        original_data["insulator_length"].to_numpy(),
    )


# ── has_manipulations ─────────────────────────────────────────────────────


def test_has_manipulations_false_initially(
    manipulation: Manipulation,
) -> None:
    assert not manipulation.has_manipulations


def test_has_manipulations_after_support(
    manipulation: Manipulation,
) -> None:
    manipulation.support_manipulation({0: {"z": 1.0}})
    assert manipulation.has_manipulations


def test_has_manipulations_after_rope(
    manipulation: Manipulation,
) -> None:
    manipulation.rope_manipulation({0: 2.0})
    assert manipulation.has_manipulations


def test_has_manipulations_after_virtual_support(
    manipulation: Manipulation,
) -> None:
    manipulation.add_virtual_support(
        {
            1: {
                "x": 100.0,
                "y": 0.0,
                "z": 55.0,
                "insulator_length": 3.0,
                "insulator_mass": 500.0,
                "hanging_cable_point_from_left_support": 100.0,
            }
        }
    )
    assert manipulation.has_manipulations


def test_has_manipulations_after_shifting(
    manipulation: Manipulation,
) -> None:
    manipulation.add_cable_shifting(
        shift_support=np.array([0.0, 1.0, 0.0, 0.0])
    )
    assert manipulation.has_manipulations


# ── cable shifting ────────────────────────────────────────────────────────


def test_add_cable_shifting_default_values(
    manipulation: Manipulation,
) -> None:
    manipulation.add_cable_shifting()

    np.testing.assert_array_equal(
        manipulation.shift_support, np.zeros(4)
    )
    np.testing.assert_array_equal(
        manipulation.shortening_span, np.zeros(3)
    )


def test_add_cable_shifting_stores_values(
    manipulation: Manipulation,
) -> None:
    shifting = np.array([0.0, 1.5, 2.0, 0.0])
    shortening = np.array([0.0, 0.5, 1.0])

    manipulation.add_cable_shifting(
        shift_support=shifting,
        shorten_span=shortening,
    )

    np.testing.assert_array_equal(
        manipulation.shift_support, shifting
    )
    np.testing.assert_array_equal(
        manipulation.shortening_span, shortening
    )


def test_add_cable_shifting_wrong_size_shifting(
    manipulation: Manipulation,
) -> None:
    with pytest.raises(ValueError):
        manipulation.add_cable_shifting(
            shift_support=np.array([0.0, 1.0, 0.0])  # 3 elements, 4 expected
        )


def test_add_cable_shifting_wrong_size_shortening(
    manipulation: Manipulation,
) -> None:
    with pytest.raises(ValueError):
        manipulation.add_cable_shifting(
            shorten_span=np.array(
                [0.0, 1.0, 0.0, 0.0]
            )  # 4 elements, 3 expected
        )


def test_add_cable_shifting_enforces_shifting_boundaries(
    manipulation: Manipulation,
) -> None:
    with pytest.warns(BalanceEngineWarning):
        manipulation.add_cable_shifting(
            shift_support=np.array([5.0, 1.0, 2.0, 3.0])
        )

    assert abs(manipulation.shift_support[0]) < 1e-5
    assert abs(manipulation.shift_support[-1]) < 1e-5
    np.testing.assert_array_equal(
        manipulation.shift_support[1:-1],
        np.array([1.0, 2.0]),
    )


def test_add_cable_shifting_no_warning_when_boundaries_are_compliant(
    manipulation: Manipulation,
) -> None:
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", BalanceEngineWarning)
        manipulation.add_cable_shifting(
            shift_support=np.array([0.0, 1.0, 2.0, 0.0]),
            shorten_span=np.array([0.0, 1.0, 2.0]),
        )


def test_has_shifting_false_initially(
    manipulation: Manipulation,
) -> None:
    assert not manipulation.has_shifting


def test_has_shifting_after_add(
    manipulation: Manipulation,
) -> None:
    manipulation.add_cable_shifting(
        shift_support=np.array([0.0, 1.0, 0.0, 0.0])
    )
    assert manipulation.has_shifting


def test_reset_cable_shifting(
    manipulation: Manipulation,
) -> None:
    manipulation.add_cable_shifting(
        shift_support=np.array([0.0, 1.0, 0.0, 0.0])
    )
    manipulation.reset_cable_shifting()
    assert not manipulation.has_shifting
    assert manipulation.shift_support is None
    assert manipulation.shortening_span is None


def test_reset_cable_shifting_no_prior(
    manipulation: Manipulation,
) -> None:
    manipulation.reset_cable_shifting()  # should not raise


def test_compute_shifted_L_ref(
    manipulation: Manipulation,
) -> None:
    initial_L_ref = np.array([500.0, 300.0, 400.0])

    # shift support 1 by 1m → span 0 gains 1m, span 1 loses 1m
    manipulation.add_cable_shifting(
        shift_support=np.array([0.0, 1.0, 0.0, 0.0])
    )
    shifted = manipulation.compute_shifted_L_ref(initial_L_ref)
    np.testing.assert_allclose(
        shifted, np.array([501.0, 299.0, 400.0])
    )


def test_compute_shifted_L_ref_with_shortening(
    manipulation: Manipulation,
) -> None:
    initial_L_ref = np.array([500.0, 300.0, 400.0])

    manipulation.add_cable_shifting(
        shorten_span=np.array([0.0, 2.0, 0.0])
    )
    shifted = manipulation.compute_shifted_L_ref(initial_L_ref)
    np.testing.assert_allclose(
        shifted, np.array([500.0, 298.0, 400.0])
    )


def test_compute_shifted_L_ref_no_shifting(
    manipulation: Manipulation,
) -> None:
    initial_L_ref = np.array([500.0, 300.0, 400.0])
    result = manipulation.compute_shifted_L_ref(initial_L_ref)
    np.testing.assert_array_equal(result, initial_L_ref)
