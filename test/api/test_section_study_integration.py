# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

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


def _dxdydz(study: SectionStudy) -> np.ndarray:
    return study._balance_engine.balance_model.nodes.dxdydz.copy()


@pytest.mark.integration
def test_move_load_between_spans_reflects_change(
    study_8span: SectionStudy,
) -> None:
    """Bug A regression: moving a load to a different span (same load count)
    must change node geometry, not be masked by a stale warm-start memento.
    """
    n_spans = len(study_8span._section_array.data.span_length) - 1
    study_8span.solve_adjustment()

    load_position = np.zeros(n_spans)
    load_mass = np.zeros(n_spans)
    load_position[1] = 100.0
    load_mass[1] = 500.0

    study_8span.set_loads(load_position, load_mass)
    study_8span.solve_change_state(new_temperature=30.0)
    geometry_load_on_span_1 = _dxdydz(study_8span)

    # move the load from span index 1 to span index 4
    load_position_moved = np.zeros(n_spans)
    load_mass_moved = np.zeros(n_spans)
    load_position_moved[4] = 100.0
    load_mass_moved[4] = 500.0

    study_8span.set_loads(load_position_moved, load_mass_moved)
    study_8span.solve_change_state(new_temperature=30.0)
    geometry_load_on_span_4 = _dxdydz(study_8span)

    assert not np.allclose(
        geometry_load_on_span_1, geometry_load_on_span_4
    ), "Moving the load between spans should change node geometry"


@pytest.mark.integration
def test_load_count_transition_1_to_2_matches_fresh_build(
    study_8span: SectionStudy, cable_array_AM600: CableArray
) -> None:
    """Bug B regression: going from 1 loaded span to 2 must not crash and
    must match a fresh SectionStudy built directly with 2 loaded spans.
    """
    n_spans = len(study_8span._section_array.data.span_length) - 1
    study_8span.solve_adjustment()

    load_position = np.zeros(n_spans)
    load_mass = np.zeros(n_spans)
    load_position[1] = 100.0
    load_mass[1] = 500.0
    study_8span.set_loads(load_position, load_mass)
    study_8span.solve_change_state(new_temperature=30.0)

    # transition to 2 loaded spans: must not crash
    load_position_2 = load_position.copy()
    load_mass_2 = load_mass.copy()
    load_position_2[5] = 150.0
    load_mass_2[5] = 300.0
    study_8span.set_loads(load_position_2, load_mass_2)
    study_8span.solve_change_state(new_temperature=30.0)
    geometry_after_transition = _dxdydz(study_8span)

    # fresh build directly with the 2-load layout
    fresh_study = SectionStudy(
        cable_array=cable_array_AM600,
        section_array=study_8span._section_array,
    )
    fresh_study.solve_adjustment()
    fresh_study.set_loads(load_position_2, load_mass_2)
    fresh_study.solve_change_state(new_temperature=30.0)
    geometry_fresh = _dxdydz(fresh_study)

    np.testing.assert_allclose(
        geometry_after_transition, geometry_fresh, atol=1e-6
    )


@pytest.mark.integration
def test_load_count_transition_2_to_1_matches_fresh_build(
    study_8span: SectionStudy, cable_array_AM600: CableArray
) -> None:
    """Bug B regression: going from 2 loaded spans to 1 must not crash and
    must match a fresh SectionStudy built directly with 1 loaded span.
    """
    n_spans = len(study_8span._section_array.data.span_length) - 1
    study_8span.solve_adjustment()

    load_position_2 = np.zeros(n_spans)
    load_mass_2 = np.zeros(n_spans)
    load_position_2[1] = 100.0
    load_mass_2[1] = 500.0
    load_position_2[5] = 150.0
    load_mass_2[5] = 300.0
    study_8span.set_loads(load_position_2, load_mass_2)
    study_8span.solve_change_state(new_temperature=30.0)

    # transition to 1 loaded span: must not crash
    load_position_1 = np.zeros(n_spans)
    load_mass_1 = np.zeros(n_spans)
    load_position_1[1] = 100.0
    load_mass_1[1] = 500.0
    study_8span.set_loads(load_position_1, load_mass_1)
    study_8span.solve_change_state(new_temperature=30.0)
    geometry_after_transition = _dxdydz(study_8span)

    fresh_study = SectionStudy(
        cable_array=cable_array_AM600,
        section_array=study_8span._section_array,
    )
    fresh_study.solve_adjustment()
    fresh_study.set_loads(load_position_1, load_mass_1)
    fresh_study.solve_change_state(new_temperature=30.0)
    geometry_fresh = _dxdydz(fresh_study)

    np.testing.assert_allclose(
        geometry_after_transition, geometry_fresh, atol=1e-6
    )


@pytest.mark.integration
def test_climate_only_repeated_solve_change_state_moves_geometry(
    study_8span: SectionStudy,
) -> None:
    """Climate-only repeated solve_change_state calls (no load changes)
    should still move geometry each time (no regression on warm-start reuse).
    """
    study_8span.solve_adjustment()

    study_8span.solve_change_state(wind_pressure=200.0)
    geometry_wind = _dxdydz(study_8span)

    study_8span.solve_change_state(new_temperature=60.0)
    geometry_temp = _dxdydz(study_8span)

    assert not np.allclose(
        geometry_wind, geometry_temp
    ), "Repeated climate-only solve_change_state calls should change geometry"


@pytest.mark.integration
def test_manipulation_registration_resets_stale_intermediate_memento(
    study_8span: SectionStudy,
) -> None:
    """Optional follow-up regression: registering a manipulation rewires the
    balance engine and caretaker in ``solve_adjustment``, but must also
    invalidate any previously stored ``_intermediate_memento``. Otherwise a
    later ``solve_change_state`` restores a memento snapshot taken from the
    old (pre-manipulation) engine, whose arrays no longer match the shape of
    the newly rewired (manipulated) engine.
    """
    # 1. Solve on the clean geometry and run a non-default climate solve so
    #    that an intermediate warm-start memento gets stored for the
    #    *pre-manipulation* engine.
    study_8span.solve_adjustment()
    study_8span.solve_change_state(new_temperature=90.0)
    assert study_8span.intermediate_memento is not None

    # 2. Register a manipulation that changes the engine shape (adds a
    #    virtual support, growing the number of nodes from 9 to 10) and
    #    re-run solve_adjustment: this rewires _balance_engine/_caretaker.
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

    # 3. A subsequent non-default solve_change_state must not attempt to
    #    restore the stale, wrong-shaped memento from step 1.
    study_8span.solve_change_state(wind_pressure=150.0)
