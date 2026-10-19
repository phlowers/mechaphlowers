# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import warnings as _warnings
from typing import cast

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from mechaphlowers.config import options
from mechaphlowers.core.geometry.line_angles import get_edge_arm_coords
from mechaphlowers.core.manipulation import Manipulation
from mechaphlowers.core.models.balance.span_loads import SpanLoads
from mechaphlowers.core.models.cable.span import ISpan
from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.entities.errors import (
    BalanceEngineWarning,
    InvalidManipulationIndex,
    InvalidManipulationKeys,
    InvalidManipulationRange,
)


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

    manipulation.modify_support({1: {"z": 3.0, "y": -2.0}})
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

    manipulation.modify_support({0: {"z": 1.0}, 2: {"z": -0.5, "y": 2.0}})
    applied = manipulation.from_section_array(section_array)

    assert_allclose(
        applied.data["conductor_attachment_altitude"].iloc[0],
        original_alt.iloc[0] + 1.0,
    )
    assert_allclose(
        applied.data["conductor_attachment_altitude"].iloc[2],
        original_alt.iloc[2] - 0.5,
    )
    assert_allclose(
        applied.data["crossarm_length"].iloc[2],
        section_array._data["crossarm_length"].iloc[2] + 2.0,
    )


def test_support_manipulation_partial_keys(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_alt = section_array._data["conductor_attachment_altitude"].copy()
    original_arm = section_array._data["crossarm_length"].copy()

    manipulation.modify_support({0: {"z": 1.5}})
    applied = manipulation.from_section_array(section_array)

    assert_allclose(
        applied.data["conductor_attachment_altitude"].iloc[0],
        original_alt.iloc[0] + 1.5,
    )
    assert_allclose(
        applied.data["crossarm_length"].iloc[0],
        original_arm.iloc[0],
    )

    manipulation.reset_support()
    manipulation.modify_support({0: {"y": -1.0}})
    applied = manipulation.from_section_array(section_array)

    assert_allclose(
        applied.data["conductor_attachment_altitude"].iloc[0],
        original_alt.iloc[0],
    )
    assert_allclose(
        applied.data["crossarm_length"].iloc[0],
        original_arm.iloc[0] - 1.0,
    )


def test_support_manipulation_add_successive_without_reset(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_alt = section_array._data["conductor_attachment_altitude"].copy()
    original_arm = section_array._data["crossarm_length"].copy()

    manipulation.modify_support({0: {"z": 1.5}})
    applied = manipulation.from_section_array(section_array)

    assert_allclose(
        applied.data["conductor_attachment_altitude"].iloc[0],
        original_alt.iloc[0] + 1.5,
    )
    assert_allclose(
        applied.data["crossarm_length"].iloc[0],
        original_arm.iloc[0],
    )

    manipulation.modify_support({0: {"y": -1.0}})
    applied = manipulation.from_section_array(section_array)
    assert_allclose(
        applied.data["conductor_attachment_altitude"].iloc[0],
        original_alt.iloc[0] + 1.5,
    )
    assert_allclose(
        applied.data["crossarm_length"].iloc[0],
        original_arm.iloc[0] - 1.0,
    )


def test_support_manipulation_invalid_index(
    manipulation: Manipulation,
) -> None:
    with pytest.raises(InvalidManipulationIndex):
        manipulation.modify_support({10: {"z": 1.0}})


def test_support_manipulation_invalid_keys(
    manipulation: Manipulation,
) -> None:
    with pytest.raises(InvalidManipulationKeys):
        manipulation.modify_support({0: {"x": 1.0}})


def test_reset_manipulation(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_alt = section_array.data["conductor_attachment_altitude"].copy()
    original_arm = section_array.data["crossarm_length"].copy()

    manipulation.modify_support({0: {"z": 5.0}, 1: {"y": -3.0}})
    manipulation.modify_support({2: {"z": 1.0}})
    manipulation.reset_support()
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

    manipulation.reset_support()  # should not raise
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

    manipulation.set_rope_or_chain(
        {
            1: {"type": "rope", "length": 5.0},
            2: {"type": "rope", "length": 3.0},
        }
    )
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

    # Original _data must be unchanged
    assert_allclose(
        section_array._data["insulator_length"].to_numpy(),
        original_insulator_length.to_numpy(),
    )
    assert_allclose(
        section_array._data["insulator_mass"].to_numpy(),
        original_insulator_mass.to_numpy(),
    )


def test_chain_manipulation_overrides_data(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_insulator_length = section_array._data["insulator_length"].copy()
    original_insulator_mass = section_array._data["insulator_mass"].copy()

    manipulation.set_rope_or_chain(
        {
            1: {"type": "chain", "length": 5.0, "mass": 45},
            2: {"type": "chain", "length": 3.0, "mass": 55},
        }
    )
    applied = manipulation.from_section_array(section_array)

    # Modified supports
    assert_allclose(applied.data["insulator_length"].iloc[1], 5.0)
    assert_allclose(applied.data["insulator_length"].iloc[2], 3.0)
    assert_allclose(
        applied.data["insulator_mass"].iloc[1], 45
    )  # default lineic mass
    assert_allclose(applied.data["insulator_mass"].iloc[2], 55)

    # Unlisted support unchanged
    assert_allclose(
        applied.data["insulator_length"].iloc[0],
        original_insulator_length.iloc[0],
    )
    assert_allclose(
        applied.data["insulator_mass"].iloc[0],
        original_insulator_mass.iloc[0],
    )

    # Original _data must be unchanged
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
    manipulation.set_rope_or_chain(
        {0: {"type": "rope", "length": 2.0, "lineic_mass": 0.5}}
    )
    applied = manipulation.from_section_array(section_array)

    assert_allclose(applied.data["insulator_mass"].iloc[0], 2.0 * 0.5)


def test_rope_manipulation_insulator_weight_updated(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    """insulator_weight in .data must reflect the rope mass."""
    manipulation.set_rope_or_chain(
        {1: {"type": "rope", "length": 4.0, "lineic_mass": 0.1}}
    )
    applied = manipulation.from_section_array(section_array)

    expected_weight = 4.0 * 0.1 * 9.81  # approx N
    assert_allclose(
        applied.data["insulator_weight"].iloc[1], expected_weight, rtol=1e-3
    )


def test_rope_manipulation_invalid_index(manipulation: Manipulation) -> None:
    with pytest.raises(InvalidManipulationIndex, match="out of range"):
        manipulation.set_rope_or_chain({99: {"type": "rope", "length": 5.0}})


def test_rope_overlay_without_lineic_mass_is_skipped(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    manipulation._rope_overlay = {1: 5.0}
    applied = manipulation.from_section_array(section_array)
    assert_allclose(
        applied.data["insulator_length"].to_numpy(),
        section_array.data["insulator_length"].to_numpy(),
    )


def test_reset_rope_manipulation(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_length = section_array.data["insulator_length"].copy()
    original_mass = section_array.data["insulator_mass"].copy()

    manipulation.set_rope_or_chain(
        {
            1: {"type": "rope", "length": 5.0},
            2: {"type": "rope", "length": 3.0},
        }
    )
    manipulation.reset_rope()
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
    manipulation.reset_rope()  # should not raise


def _arm_altitude(sa: SectionArray) -> np.ndarray:
    data = sa.data
    center_arm_coords, _ = get_edge_arm_coords(
        np.zeros((len(data), 3)),
        data["conductor_attachment_altitude"].to_numpy(),
        data["crossarm_length"].to_numpy(),
        data["line_angle"].to_numpy(),
        data["insulator_length"].to_numpy(),
    )
    return center_arm_coords[:, 2]


def test_set_rope_keeps_arm_attachment_fixed(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_arm_altitude = _arm_altitude(section_array)

    manipulation.set_rope({1: 5.0, 2: 1.0})
    applied = manipulation.from_section_array(section_array)

    assert_allclose(_arm_altitude(applied), original_arm_altitude)


def test_set_rope_moves_cable_attachment(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    original_alt = section_array.data["conductor_attachment_altitude"]
    original_length = section_array.data["insulator_length"]

    manipulation.set_rope({1: 5.0, 2: 1.0})
    applied = manipulation.from_section_array(section_array)

    # arm stays, so the cable hangs at arm_altitude - rope_length
    assert_allclose(
        applied.data["conductor_attachment_altitude"].to_numpy()[1:3],
        [
            original_alt.iloc[1] + original_length.iloc[1] - 5.0,
            original_alt.iloc[2] + original_length.iloc[2] - 1.0,
        ],
    )


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
    manip.modify_support({1: {"z": 2.0}})
    applied = manip.from_section_array(sa)
    data = applied.data

    # Only support 1 is affected
    assert_allclose(data["counterweight"].iloc[1], 0.0)
    # Other supports with counterweight are unchanged
    assert_allclose(
        data["counterweight"].iloc[2], original_counterweight.iloc[2]
    )

    manip.reset_support()
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
    manip.set_rope_or_chain({1: {"type": "rope", "length": 4.5}})
    applied = manip.from_section_array(sa)
    data = applied.data

    # Only support 1 is affected
    assert_allclose(data["counterweight"].iloc[1], 0.0)
    # Other supports with counterweight are unchanged
    assert_allclose(
        data["counterweight"].iloc[2], original_counterweight.iloc[2]
    )

    manip.reset_rope()
    applied = manip.from_section_array(sa)
    assert_allclose(
        applied.data["counterweight"].to_numpy(),
        original_counterweight.to_numpy(),
    )


# ── virtual support ──────────────────────────────────────────────────────


def test_set_virtual_support_changes_data_shape(
    section_array: SectionArray,
) -> None:
    manip = Manipulation(section_array)
    assert len(section_array.data) == 4
    manip.set_virtual_support(
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


def _virtual_support(x: float) -> dict[str, float]:
    return {
        "x": x,
        "y": 0.0,
        "z": 55.0,
        "insulator_length": 3.0,
        "insulator_mass": 500.0,
        "hanging_cable_point_from_left_support": x,
    }


@pytest.mark.parametrize("span_index", [-1, 3])
def test_set_virtual_support_invalid_index(
    manipulation: Manipulation, span_index: int
) -> None:
    virtual_support = _virtual_support(100.0)
    with pytest.raises(InvalidManipulationIndex, match="out of range"):
        manipulation.set_virtual_support({span_index: virtual_support})


def test_set_virtual_support_missing_keys(
    manipulation: Manipulation,
) -> None:
    virtual_support = _virtual_support(100.0)
    del virtual_support["z"]
    with pytest.raises(InvalidManipulationKeys, match="Missing keys"):
        manipulation.set_virtual_support({1: virtual_support})


def test_set_virtual_support_x_out_of_range(
    manipulation: Manipulation,
) -> None:
    virtual_support = _virtual_support(600.0)
    with pytest.raises(InvalidManipulationRange, match="x=.*out of range"):
        manipulation.set_virtual_support({1: virtual_support})


def test_set_virtual_support_hanging_point_out_of_range(
    manipulation: Manipulation,
) -> None:
    virtual_support = _virtual_support(100.0)
    virtual_support["hanging_cable_point_from_left_support"] = 600.0
    with pytest.raises(
        InvalidManipulationRange,
        match="hanging_cable_point_from_left_support=.*out of range",
    ):
        manipulation.set_virtual_support({1: virtual_support})


def test_reset_virtual_support_no_prior(manipulation: Manipulation) -> None:
    manipulation.reset_virtual_support()
    assert manipulation._virtual_support_overlay is None


def test_compute_split_l_ref_without_virtual_support(
    manipulation: Manipulation,
) -> None:
    initial_l_ref = np.array([10.0, 20.0, 30.0])
    result = manipulation.compute_split_L_ref(
        initial_l_ref, cast(ISpan, object())
    )
    assert result is initial_l_ref


def test_apply_virtual_support_overlay_without_overlay(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    raw_data = section_array._data.copy()
    assert manipulation._apply_virtual_support_overlay(raw_data) is raw_data


def test_apply_virtual_support_overlay_sets_ground_altitude(
    section_array: SectionArray,
    manipulation: Manipulation,
) -> None:
    raw_data = section_array._data.copy()
    raw_data["ground_altitude"] = [0.0, 0.0, 0.0, 0.0]
    manipulation._virtual_support_overlay = {1: _virtual_support(100.0)}

    applied = manipulation._apply_virtual_support_overlay(raw_data)

    assert len(applied) == 5
    assert applied.loc[2, "name"] == "virtual_1"
    assert applied.loc[2, "ground_altitude"] == (
        55.0 - options.ground.default_support_length
    )


def test_span_index_after_virtual_support_no_overlay(
    manipulation: Manipulation,
) -> None:
    assert [
        manipulation.span_index_after_virtual_support(i) for i in range(3)
    ] == [0, 1, 2]


def test_span_index_after_virtual_support_matches_span_loads(
    manipulation: Manipulation,
) -> None:
    manipulation.set_virtual_support({1: _virtual_support(100.0)})
    span_length = manipulation._section_array.data.span_length.to_numpy()
    for i in (0, 2):
        positions = np.zeros(3)
        masses = np.zeros(3)
        positions[i] = 0.5
        masses[i] = 10.0
        new_loads = manipulation.build_new_span_loads_virtual_support(
            SpanLoads(positions, masses, span_length)
        )
        j = manipulation.span_index_after_virtual_support(i)
        assert j is not None
        assert new_loads.load_mass[j] == 10.0
        assert np.count_nonzero(new_loads.load_mass) == 1
    assert manipulation.span_index_after_virtual_support(1) is None


def test_span_index_after_virtual_support_multiple(
    manipulation: Manipulation,
) -> None:
    manipulation.set_virtual_support(
        {0: _virtual_support(0.5), 2: _virtual_support(100.0)}
    )
    assert manipulation.span_index_after_virtual_support(0) is None
    assert manipulation.span_index_after_virtual_support(1) == 2
    assert manipulation.span_index_after_virtual_support(2) is None


def test_span_index_after_virtual_support_invalid_index(
    manipulation: Manipulation,
) -> None:
    with pytest.raises(InvalidManipulationIndex):
        manipulation.span_index_after_virtual_support(3)


def test_reset_virtual_support_restores_data_shape(
    section_array: SectionArray,
) -> None:
    manip = Manipulation(section_array)
    manip.set_virtual_support(
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

    manipulation.modify_support({0: {"z": 10.0}})
    manipulation.set_rope_or_chain({1: {"type": "rope", "length": 6}})
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
    manipulation.modify_support({0: {"z": 1.0}})
    assert manipulation.has_manipulations


def test_has_manipulations_after_rope(
    manipulation: Manipulation,
) -> None:
    manipulation.set_rope_or_chain({0: {"type": "rope", "length": 2}})
    assert manipulation.has_manipulations


def test_has_manipulations_after_virtual_support(
    manipulation: Manipulation,
) -> None:
    manipulation.set_virtual_support(
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
    manipulation.modify_cable(shift_support={1: 1.0})
    assert manipulation.has_manipulations


# ── cable shifting ────────────────────────────────────────────────────────


def test_add_cable_shifting_default_values(
    manipulation: Manipulation,
) -> None:
    manipulation.modify_cable()

    assert manipulation.shift_support is not None
    assert manipulation.shortening_span is not None
    np.testing.assert_array_equal(manipulation.shift_support, np.zeros(4))
    np.testing.assert_array_equal(manipulation.shortening_span, np.zeros(3))


def test_add_cable_shifting_stores_values(
    manipulation: Manipulation,
) -> None:
    manipulation.modify_cable(
        shift_support={1: 1.5, 2: 2.0},
        shorten_span={1: 0.5, 2: 1.0},
    )

    expected_shift = np.array([0.0, 1.5, 2.0, 0.0])
    expected_shorten = np.array([0.0, 0.5, 1.0])
    np.testing.assert_array_equal(manipulation.shift_support, expected_shift)
    np.testing.assert_array_equal(
        manipulation.shortening_span, expected_shorten
    )


def test_add_cable_shifting_wrong_size_shifting(
    manipulation: Manipulation,
) -> None:
    with pytest.raises(InvalidManipulationIndex):
        manipulation.modify_cable(
            shift_support={99: 1.0}  # index out of range
        )


def test_add_cable_shifting_wrong_size_shortening(
    manipulation: Manipulation,
) -> None:
    with pytest.raises(InvalidManipulationIndex):
        manipulation.modify_cable(
            shorten_span={99: 1.0}  # index out of range
        )


def test_add_cable_shifting_enforces_shifting_boundaries(
    manipulation: Manipulation,
) -> None:
    with pytest.warns(BalanceEngineWarning):
        manipulation.modify_cable(
            shift_support={0: 5.0, 1: 1.0, 2: 2.0, 3: 3.0}
        )

    assert manipulation.shift_support is not None
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
        manipulation.modify_cable(
            shift_support={1: 1.0, 2: 2.0},
            shorten_span={0: 0.0, 1: 1.0, 2: 2.0},
        )


def test_has_shifting_false_initially(
    manipulation: Manipulation,
) -> None:
    assert not manipulation.has_shifting


def test_has_shifting_after_add(
    manipulation: Manipulation,
) -> None:
    manipulation.modify_cable(shift_support={1: 1.0})
    assert manipulation.has_shifting


def test_reset_cable_shifting(
    manipulation: Manipulation,
) -> None:
    manipulation.modify_cable(shift_support={1: 1.0})
    manipulation.reset_cable()
    assert not manipulation.has_shifting
    assert manipulation.shift_support is None
    assert manipulation.shortening_span is None


def test_reset_cable_shifting_no_prior(
    manipulation: Manipulation,
) -> None:
    manipulation.reset_cable()  # should not raise


def test_compute_shifted_L_ref(
    manipulation: Manipulation,
) -> None:
    initial_L_ref = np.array([500.0, 300.0, 400.0])

    # shift support 1 by 1m → span 0 gains 1m, span 1 loses 1m
    manipulation.modify_cable(shift_support={1: 1.0})
    shifted = manipulation.compute_shifted_L_ref(initial_L_ref)
    np.testing.assert_allclose(shifted, np.array([501.0, 299.0, 400.0]))


def test_compute_shifted_L_ref_with_shortening(
    manipulation: Manipulation,
) -> None:
    initial_L_ref = np.array([500.0, 300.0, 400.0])

    manipulation.modify_cable(shorten_span={1: 2.0})
    shifted = manipulation.compute_shifted_L_ref(initial_L_ref)
    np.testing.assert_allclose(shifted, np.array([500.0, 298.0, 400.0]))


def test_compute_shifted_L_ref_no_shifting(
    manipulation: Manipulation,
) -> None:
    initial_L_ref = np.array([500.0, 300.0, 400.0])
    result = manipulation.compute_shifted_L_ref(initial_L_ref)
    np.testing.assert_array_equal(result, initial_L_ref)
