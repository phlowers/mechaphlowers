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


@pytest.fixture
def study_8span(cable_array_AM600: CableArray) -> SectionStudy:
    """8-span section (9 supports) with one line angle at support 4."""
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
                "suspension": [
                    False, True, True, True, True, True, True, True, False
                ],
                "conductor_attachment_altitude": [
                    30.0, 45.0, 55.0, 60.0, 50.0, 65.0, 40.0, 55.0, 35.0
                ],
                "crossarm_length": [
                    0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0
                ],
                "line_angle": [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "insulator_length": [
                    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0
                ],
                "span_length": [
                    400.0, 350.0, 450.0, 300.0, 500.0, 380.0, 420.0, 360.0,
                    np.nan,
                ],
                "insulator_mass": [
                    1000.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0,
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
    return SectionStudy(cable_array=cable_array_AM600, section_array=section_array)


@pytest.mark.integration
def test_lengthen(study_8span: SectionStudy) -> None:
    """Negative shortening lengthens spans."""
    study_8span.manipulation.shift_cable(
        shift_support=[0.0, -2.0, 0.0, -1.5, 0.0, 0.0, -3.0, 0.0, 0.0],
    )
    study_8span.solve_adjustment()
    study_8span.solve_change_state(new_temperature=15.0)
    assert True


@pytest.mark.integration
def test_shorten(study_8span: SectionStudy) -> None:
    """Positive shortening shortens spans."""
    study_8span.manipulation.shift_cable(
        shorten_span=[0.0, 2.0, 0.0, 1.5, 0.0, 0.0, 3.0, 0.0],
    )
    study_8span.solve_adjustment()
    study_8span.solve_change_state(new_temperature=15.0)
    assert True


@pytest.mark.integration
def test_cable_shifting(study_8span: SectionStudy) -> None:
    """Cable shifting with horizontal offsets."""
    study_8span.manipulation.shift_cable(
        shift_support=[0.0, 1.0, -0.5, 2.0, 0.0, -1.0, 0.5, 1.5, 0.0],
        shorten_span=[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    study_8span.solve_adjustment()
    study_8span.solve_change_state(new_temperature=15.0)
    assert True


@pytest.mark.integration
def test_rope(study_8span: SectionStudy) -> None:
    """Replace insulators with rope on two supports."""
    study_8span.manipulation.rope_manipulation({2: 5.0, 5: 4.0})
    study_8span.solve_adjustment()
    study_8span.solve_change_state(new_temperature=15.0)
    assert True


@pytest.mark.integration
def test_support_shifting(study_8span: SectionStudy) -> None:
    """Shift supports vertically and laterally."""
    study_8span.manipulation.support_manipulation({
        1: {"z": 2.0},
        3: {"z": -1.5, "y": 1.0},
        6: {"z": 3.0},
    })
    study_8span.solve_adjustment()
    study_8span.solve_change_state(new_temperature=15.0)
    assert True


@pytest.mark.integration
def test_virtual_support(study_8span: SectionStudy) -> None:
    """Add a virtual support in span 4 (longest span: 500m)."""
    
    study_8span.solve_adjustment()  # solve before adding virtual support to compute initial L_ref
    print(study_8span.balance_engine.L_ref)
    study_8span.manipulation.add_virtual_support({
        4: {
            "x": 250.0,
            "y": 0.0,
            "z": 45.0,
            "insulator_length": 3.0,
            "insulator_mass": 500.0,
            "hanging_cable_point_from_left_support": 250.0,
        },
        # 5: {
        #     "x": 50.0,
        #     "y": 0.0,
        #     "z": 45.0,
        #     "insulator_length": 3.0,
        #     "insulator_mass": 500.0,
        #     "hanging_cable_point_from_left_support": 50.0,
        # }
    })
    study_8span.solve_adjustment()
    print(study_8span.balance_engine.L_ref)
    study_8span.solve_change_state(new_temperature=15.0)
    assert True


def test_section_array_with_virtual_support(cable_array_AM600):

    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4", "5", "virtual_4", "6", "7", "8", "9"],
                "suspension": [
                    False, True, True, True, True, True, True, True, True, False
                ],
                "conductor_attachment_altitude": [
                    30.0, 45.0, 55.0, 60.0, 50.0, 45.0, 65.0, 40.0, 55.0, 35.0
                ],
                "crossarm_length": [
                    0.0, 5.0, 5.0, 5.0, 5.0, 0.0, 5.0, 5.0, 5.0, 0.0
                ],
                "line_angle": [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "insulator_length": [
                    3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0
                ],
                "span_length": [
                    400.0, 350.0, 450.0, 300.0, 250., 250.0, 380.0, 420.0, 360.0,
                    np.nan,
                ],
                "insulator_mass": [
                    1000.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0,
                    1000.0,
                ],
                "load_mass": [0.0] * 10,
                "load_position": [0.0] * 10,
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    ss = SectionStudy(cable_array=cable_array_AM600, section_array=section_array)

    ss.solve_adjustment()
    ss.solve_change_state(new_temperature=15.0)