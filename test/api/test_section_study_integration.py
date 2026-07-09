# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import copy

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.api.section_study import SectionStudy
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.entities.errors import SolverError


@pytest.fixture
def study_8span(cable_array_AM600: CableArray) -> SectionStudy:
    """8-span section (9 supports) with one line angle at support 4."""
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
                "suspension": [
                    False,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
                "conductor_attachment_altitude": [
                    30.0,
                    45.0,
                    55.0,
                    60.0,
                    50.0,
                    65.0,
                    40.0,
                    55.0,
                    35.0,
                ],
                "crossarm_length": [
                    0.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    5.0,
                    0.0,
                ],
                "line_angle": [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "insulator_length": [
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                    3.0,
                ],
                "span_length": [
                    400.0,
                    350.0,
                    450.0,
                    300.0,
                    500.0,
                    380.0,
                    420.0,
                    360.0,
                    np.nan,
                ],
                "insulator_mass": [
                    1000.0,
                    500.0,
                    500.0,
                    500.0,
                    500.0,
                    500.0,
                    500.0,
                    500.0,
                    1000.0,
                ],
                "load_mass": [0.0] * 9,
                "load_position": [0.0] * 9,
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    return SectionStudy(
        cable_array=cable_array_AM600, section_array=section_array
    )


@pytest.fixture
def study_8span_Lref(study_8span: SectionStudy) -> np.ndarray:
    """Reference lengths for the 8 spans in the study_8span."""

    study_8span.solve_adjustment()
    return study_8span._balance_engine.L_ref


@pytest.mark.integration
def test_lengthen(
    study_8span: SectionStudy, study_8span_Lref: np.ndarray
) -> None:
    """Negative shortening lengthens spans."""
    study_8span.manipulation.modify_cable(
        shift_support={1: 2.0, 3: -1.5, 6: -3.0},
    )
    study_8span.solve_adjustment()
    study_8span.solve_change_state(new_temperature=15.0)

    expected_shift = np.array(
        [
            2.0,
            -2.0,
            -1.5,
            1.5,
            0.0,
            -3.0,
            3.0,
            0.0,
        ]
    )

    np.testing.assert_allclose(
        study_8span._balance_engine.L_ref,
        study_8span_Lref + expected_shift,
    )

    np.testing.assert_allclose(
        study_8span.manipulation.compute_shifted_L_ref(
            np.zeros_like(study_8span_Lref)
        ),
        expected_shift,
    )


@pytest.mark.integration
def test_shorten(
    study_8span: SectionStudy, study_8span_Lref: np.ndarray
) -> None:
    """Positive shortening shortens spans."""
    study_8span.manipulation.modify_cable(
        shorten_span={1: 2.0, 3: -1.5, 6: -3.0},
    )
    study_8span.solve_adjustment()
    study_8span.solve_change_state(new_temperature=15.0)

    expected_shorten = np.array(
        [
            0.0,
            2.0,
            0.0,
            -1.5,
            0.0,
            0.0,
            -3.0,
            0.0,
        ]
    )
    np.testing.assert_allclose(
        study_8span._balance_engine.L_ref,
        study_8span_Lref - expected_shorten,
    )

    assert study_8span.manipulation.shortening_span is not None
    np.testing.assert_allclose(
        study_8span.manipulation.shortening_span,
        expected_shorten,
    )


@pytest.mark.integration
def test_cable_shifting(
    study_8span: SectionStudy, study_8span_Lref: np.ndarray
) -> None:
    """Cable shifting with horizontal offsets."""
    study_8span.manipulation.modify_cable(
        shift_support={
            1: 1.0,
            2: -0.5,
            3: 2.0,
            5: -1.0,
        },
        shorten_span={1: 2.0},
    )
    study_8span.solve_adjustment()
    study_8span.solve_change_state(new_temperature=15.0)
    print(study_8span._balance_engine.L_ref)
    expected_shorten_shift = np.array(
        [
            -1.0,
            1.0 + 0.5 + 2,
            -0.5 - 2,
            2,
            1.0,
            -1.0,
            0.0,
            0.0,
        ]
    )
    np.testing.assert_allclose(
        study_8span._balance_engine.L_ref,
        study_8span_Lref - expected_shorten_shift,
    )


@pytest.mark.integration
def test_rope(study_8span: SectionStudy) -> None:
    """Replace insulators with rope on two supports."""
    study_8span.manipulation.add_rope({2: 5.0, 5: 4.0})
    study_8span.solve_adjustment()
    study_8span.solve_change_state(new_temperature=15.0)

    expected_insulator_length = np.array(
        [3.0, 3.0, 5.0, 3.0, 3.0, 4.0, 3.0, 3.0, 3.0]
    )
    np.testing.assert_allclose(
        study_8span.balance_engine.section_array.data["insulator_length"],
        expected_insulator_length,
    )


@pytest.mark.integration
def test_intermediate_state_should_refresh_after_load_addition(
    study_8span: SectionStudy,
) -> None:
    """Add a load on support 5 and check that the intermediate state is refreshed."""
    study_8span.solve_adjustment()
    study_8span.solve_change_state(new_temperature=15.0)

    # intermediate_memento_0 = copy.copy(study_8span._intermediate_memento)

    expected_load_mass = np.array([0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0])
    expected_load_position = np.array(
        [0.0, 0.0, 0.0, 0.0, 250.0, 0.0, 0.0, 0.0]
    )

    study_8span.set_loads(
        load_position_distance=expected_load_position,
        load_mass=expected_load_mass,
    )
    intermediate_memento_1 = copy.copy(study_8span._intermediate_memento)
    assert intermediate_memento_1 is not None
    np.testing.assert_allclose(
        intermediate_memento_1.load_mass, expected_load_mass
    )


@pytest.mark.integration
def test_reset_engine_breaks_reactivity_on_position_engine_after_climate_load_addition(
    study_8span: SectionStudy,
) -> None:
    """Add a load on support 5 and check that the intermediate state is refreshed."""

    # very important: full reset
    study_8span.balance_engine.reset(True)

    # 0. solve adjustment and change state to initialize the intermediate state
    study_8span.solve_adjustment()
    study_8span.solve_change_state(new_temperature=15.0)
    group_0 = study_8span.position_engine.get_group_points()

    # 1. change state to a new temperature and wind pressure, which will break the intermediate state
    study_8span.solve_change_state(new_temperature=15.0, wind_pressure=300)
    group_1 = study_8span.position_engine.get_group_points()

    # 2. add a load
    expected_load_mass = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected_load_position = np.array(
        [250.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    study_8span.set_loads(
        load_position_distance=expected_load_position,
        load_mass=expected_load_mass,
    )
    group_2 = study_8span.position_engine.get_group_points()

    # should be different because the intermediate state is not refreshed after load addition
    assert group_0 is not None
    assert group_1 is not None
    assert group_2 is not None
    assert group_0.spans is not None
    assert group_1.spans is not None
    assert group_2.spans is not None
    group_0_coords = group_0.spans.coords[0]
    group_1_coords = group_1.spans.coords[0]
    group_2_coords = group_2.spans.coords[0]
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(group_0_coords, group_1_coords)
        np.testing.assert_allclose(group_1_coords, group_2_coords)
        np.testing.assert_allclose(group_0_coords, group_2_coords)


@pytest.mark.integration
def test_virtual_support(study_8span: SectionStudy) -> None:
    """Add a virtual support in span 4 (longest span: 500m)."""

    study_8span.solve_adjustment()  # solve before adding virtual support to compute initial L_ref
    study_8span.manipulation.add_virtual_support(
        {
            4: {
                "x": 250.0,
                "y": 0.0,
                "z": 45.0,
                "insulator_length": 4.0,
                "insulator_mass": 495.0,
                "hanging_cable_point_from_left_support": 250.0,
            },
        }
    )
    study_8span.solve_adjustment()

    study_8span.solve_change_state(new_temperature=15.0)
    assert len(study_8span._balance_engine.L_ref) == 9
    assert (
        study_8span.balance_engine.section_array.data["span_length"][5]
        == 250.0
    )
    assert (
        study_8span.balance_engine.section_array.data[
            "conductor_attachment_altitude"
        ][5]
        == 45.0
    )
    assert (
        study_8span.balance_engine.section_array.data["insulator_length"][5]
        == 4.0
    )
    assert (
        study_8span.balance_engine.section_array.data["insulator_mass"][5]
        == 495.0
    )


@pytest.mark.integration
def test_virtual_support_doesnot_change_input_index(
    study_8span: SectionStudy, study_8span_Lref: np.ndarray
) -> None:
    """Add a virtual support in span 4 (longest span: 500m)."""

    study_8span.manipulation.modify_cable(
        shift_support={
            1: 1.0,
            6: 2.0,
            7: 1.0,
        },
    )
    study_8span.manipulation.modify_support(
        {
            5: {"z": 2.0},
        }
    )
    study_8span.add_rope({5: 4.0})
    study_8span.solve_adjustment()
    study_8span.solve_change_state(new_temperature=15.0)

    L_ref_before_support_addition = study_8span._balance_engine.L_ref.copy()

    study_8span.manipulation.add_virtual_support(
        {
            4: {
                "x": 250.0,
                "y": 0.0,
                "z": 45.0,
                "insulator_length": 3.0,
                "insulator_mass": 500.0,
                "hanging_cable_point_from_left_support": 250.0,
            },
        }
    )
    study_8span.solve_adjustment()
    np.testing.assert_allclose(
        L_ref_before_support_addition[:4],
        study_8span._balance_engine.L_ref[:4],
    )
    np.testing.assert_allclose(
        L_ref_before_support_addition[5:],
        study_8span._balance_engine.L_ref[6:],
    )

    study_8span.reset_all()  # reset manipulations to check that virtual support addition does not change input index

    # Check in the inverse affectation order
    study_8span.manipulation.add_virtual_support(
        {
            4: {
                "x": 250.0,
                "y": 0.0,
                "z": 45.0,
                "insulator_length": 3.0,
                "insulator_mass": 500.0,
                "hanging_cable_point_from_left_support": 250.0,
            },
        }
    )
    study_8span.manipulation.modify_cable(
        shift_support={
            1: 1.0,
            6: 2.0,
            7: 1.0,
        },
    )
    study_8span.manipulation.modify_support(
        {
            5: {"z": 2.0},
        }
    )
    study_8span.add_rope({5: 4.0})
    study_8span.solve_adjustment()

    study_8span.solve_adjustment()

    np.testing.assert_allclose(
        L_ref_before_support_addition[:4],
        study_8span._balance_engine.L_ref[:4],
    )
    np.testing.assert_allclose(
        L_ref_before_support_addition[5:],
        study_8span._balance_engine.L_ref[6:],
    )


@pytest.mark.integration
def test_virtual_support_and_manip_same_span(
    study_8span: SectionStudy, study_8span_Lref: np.ndarray
) -> None:
    """Add a virtual support in span 4 (longest span: 500m)."""

    study_8span.manipulation.modify_cable(
        shift_support={
            4: 1.0,
        },
    )
    study_8span.solve_adjustment()

    study_8span.solve_change_state(new_temperature=15.0)

    L_ref_before_support_addition = study_8span._balance_engine.L_ref.copy()

    study_8span.manipulation.add_virtual_support(
        {
            4: {
                "x": 250.0,
                "y": 0.0,
                "z": 45.0,
                "insulator_length": 3.0,
                "insulator_mass": 500.0,
                "hanging_cable_point_from_left_support": 250.0,
            },
        }
    )
    study_8span.solve_adjustment()
    np.testing.assert_allclose(
        L_ref_before_support_addition[:4],
        study_8span._balance_engine.L_ref[:4],
    )
    np.testing.assert_allclose(
        L_ref_before_support_addition[5:],
        study_8span._balance_engine.L_ref[6:],
    )
    study_8span.solve_change_state(new_temperature=15.0)


@pytest.mark.integration
def test_solve_change_state_wrong_array_shape_raises(
    study_8span: SectionStudy,
) -> None:
    """Passing an array with wrong shape to solve_change_state raises ValueError."""
    wrong = np.ones(5)  # 8-span section expects shape (8,)
    with pytest.raises(ValueError, match="wind_pressure"):
        study_8span.solve_change_state(wind_pressure=wrong)
    with pytest.raises(ValueError, match="ice_thickness"):
        study_8span.solve_change_state(ice_thickness=wrong)
    with pytest.raises(ValueError, match="new_temperature"):
        study_8span.solve_change_state(new_temperature=wrong)


@pytest.mark.integration
def test_rollback_with_manipulation(
    study_8span: SectionStudy, study_8span_Lref: np.ndarray
) -> None:
    """Rollback manipulations with manipulation should restore manipulated balance engine."""
    expected_L_ref_after_manipulation = study_8span_Lref + np.array(
        [
            2.0,
            -2.0,
            -1.5,
            1.5,
            0.0,
            -3.0,
            3.0,
            0.0,
        ]
    )

    study_8span.manipulation.modify_cable(
        shift_support={1: 2.0, 3: -1.5, 6: -3.0},
    )
    study_8span.manipulation.add_virtual_support(
        {
            4: {
                "x": 250.0,
                "y": 0.0,
                "z": 45.0,
                "insulator_length": 3.0,
                "insulator_mass": 500.0,
                "hanging_cable_point_from_left_support": 250.0,
            },
        }
    )
    study_8span.solve_adjustment()
    # check L_ref size changed after virtual support addition
    assert len(study_8span._balance_engine.L_ref) == 9

    with pytest.raises(SolverError):
        study_8span.solve_change_state(wind_pressure=1500)

    study_8span.solve_change_state(wind_pressure=150)
    # check L_ref size is the same as before virtual support addition, meaning that rollback has taken into account the manipulations
    assert len(study_8span._balance_engine.L_ref) == 9

    np.testing.assert_allclose(
        study_8span._balance_engine.L_ref[0:4],
        expected_L_ref_after_manipulation[0:4],
    )
