# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.core.models.guying import Guying, GuyingResults
from mechaphlowers.data.units import Q_
from mechaphlowers.entities.arrays import CableArray, SectionArray

section_array_flat = SectionArray(
    pd.DataFrame(
        {
            "name": ["1", "2", "3", "4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [30, 30, 30, 30],
            "crossarm_length": [0, 0, 0, 0],
            "line_angle": [0, 0, 0, 0],
            "insulator_length": [0.01, 3, 3, 0.01],
            "span_length": [400, 400, 400, np.nan],
            "insulator_mass": [0, 100, 100, 0],
            "load_mass": [0, 0, 0, np.nan],
            "load_position": [0.2, 0.4, 0.6, np.nan],
        }
    ),
    sagging_parameter=2000,
    sagging_temperature=15,
)
section_array_flat.add_units({"line_angle": "grad"})

section_array_span_change = SectionArray(
    pd.DataFrame(
        {
            "name": ["1", "2", "3", "4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [30, 30, 30, 30],
            "crossarm_length": [0, 0, 0, 0],
            "line_angle": [0, 0, 0, 0],
            "insulator_length": [0.01, 3, 3, 0.01],
            "span_length": [500, 300, 400, np.nan],
            "insulator_mass": [0, 100, 100, 0],
            "load_mass": [0, 0, 0, np.nan],
            "load_position": [0.2, 0.4, 0.6, np.nan],
        }
    ),
    sagging_parameter=2000,
    sagging_temperature=15,
)
section_array_span_change.add_units({"line_angle": "grad"})

section_array_complete = SectionArray(
    pd.DataFrame(
        {
            "name": ["1", "2", "3", "4"],
            "suspension": [False, True, True, False],
            "conductor_attachment_altitude": [30, 50, 60, 65],
            "crossarm_length": [0, 10, 10, 0],
            "line_angle": [0, 10, 0, 0],
            "insulator_length": [0.01, 3, 3, 0.01],
            "span_length": [500, 300, 400, np.nan],
            "insulator_mass": [100, 50, 500, 0],
            "load_mass": [0, 0, 0, np.nan],
            "load_position": [0.2, 0.4, 0.6, np.nan],
        }
    ),
    sagging_parameter=2000,
    sagging_temperature=15,
)
section_array_complete.add_units({"line_angle": "grad"})


expected_guying_left_flat = {
    "guying_tension": Q_(4119.0, "daN"),
    "vertical_force": Q_(2676.0, "daN"),
    "longitudinal_force": Q_(0.0, "daN"),
    "guying_angle_degrees": Q_(31.0, "degrees"),
}

expected_guying_pulley_left_flat = {
    "guying_tension": Q_(3549.0, "daN"),
    "vertical_force": Q_(2383.0, "daN"),
    "longitudinal_force": Q_(488.0, "daN"),
    "guying_angle_degrees": Q_(31.0, "degrees"),
}

expected_guying_left_span_change = {
    "guying_tension": Q_(4119.0, "daN"),
    "vertical_force": Q_(2585.0, "daN"),
    "longitudinal_force": Q_(0.0, "daN"),
    "guying_angle_degrees": Q_(31.0, "degrees"),
}

expected_guying_pulley_left_span_change = {
    "guying_tension": Q_(3542.0, "daN"),
    "vertical_force": Q_(2290.0, "daN"),
    "longitudinal_force": Q_(495.0, "daN"),
    "guying_angle_degrees": Q_(31.0, "degrees"),
}

expected_guying_left_complete = {
    "guying_tension": Q_(5016.0, "daN"),
    "vertical_force": Q_(3888.0, "daN"),
    "longitudinal_force": Q_(0.0, "daN"),
    "guying_angle_degrees": Q_(45.2, "degrees"),
}

expected_guying_pulley_left_complete = {
    "guying_tension": Q_(3535.0, "daN"),
    "vertical_force": Q_(2836.0, "daN"),
    "longitudinal_force": Q_(1043.0, "daN"),
    "guying_angle_degrees": Q_(45.2, "degrees"),
}


expected_guying_right_complete = {
    "guying_tension": Q_(5518.0, "daN"),
    "vertical_force": Q_(5255.0, "daN"),
    "longitudinal_force": Q_(0.0, "daN"),
    "guying_angle_degrees": Q_(50.2, "degrees"),
}

expected_guying_pulley_right_complete = {
    "guying_tension": Q_(3552.0, "daN"),
    "vertical_force": Q_(3745.0, "daN"),
    "longitudinal_force": Q_(1258.0, "daN"),
    "guying_angle_degrees": Q_(50.2, "degrees"),
}

section_array_inputs = [
    (
        section_array_flat,
        1,
        "left",
        expected_guying_left_flat,
        expected_guying_pulley_left_flat,
    ),
    (
        section_array_span_change,
        1,
        "left",
        expected_guying_left_span_change,
        expected_guying_pulley_left_span_change,
    ),
    (
        section_array_complete,
        1,
        "left",
        expected_guying_left_complete,
        expected_guying_pulley_left_complete,
    ),
    (
        section_array_complete,
        2,
        "right",
        expected_guying_right_complete,
        expected_guying_pulley_right_complete,
    ),
]


@pytest.mark.parametrize(
    "section_array, support_index, side, expected_guying_left, expected_guying_pulley_left",
    section_array_inputs,
    ids=[
        "flat_section_array_left",
        "span_change_section_array_left",
        "complete_section_array_left",
        "complete_section_array_right",
    ],
)
def test_guying_integration(
    section_array: SectionArray,
    support_index: int,
    side: Literal["left", "right"],
    expected_guying_left: dict,
    expected_guying_pulley_left: dict,
    cable_array_AM600: CableArray,
):
    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array,
    )
    balance_engine.solve_adjustment()
    balance_engine.solve_change_state(
        new_temperature=15,
        wind_pressure=0,
    )
    guying = Guying(balance_engine)

    guying_results = guying.compute(
        index=support_index,
        side=side,
        with_pulley=False,
        altitude=0,
        horizontal_distance=50,
    )

    # imposing custom tolerances for the test because of small differences with prototypes setup
    guying_results.atol_map = {
        "guying_tension": (15.0, "daN"),
        "vertical_force": (15.0, "daN"),
        "longitudinal_force": (15.0, "daN"),
        "guying_angle_degrees": (0.1, "degree"),
    }

    assert guying_results == GuyingResults(**expected_guying_left)

    guying_pulley_results = guying.compute(
        index=support_index,
        side=side,
        with_pulley=True,
        altitude=0,
        horizontal_distance=50,
    )

    # imposing custom tolerances for the test because of small differences with prototypes setup
    guying_pulley_results.atol_map = {
        "guying_tension": (15.0, "daN"),
        "vertical_force": (15.0, "daN"),
        "longitudinal_force": (15.0, "daN"),
        "guying_angle_degrees": (0.1, "degree"),
    }

    # for v1, v2 in zip(guying_pulley_results().values, GuyingResults(**expected_guying_pulley_left)().values):
    #     print(v1, v2)
    assert guying_pulley_results == GuyingResults(
        **expected_guying_pulley_left
    )


@pytest.fixture()
def guying_basic_setup(cable_array_AM600):
    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array_flat,
    )
    balance_engine.solve_adjustment()
    balance_engine.solve_change_state(
        new_temperature=15,
        wind_pressure=0,
    )
    guying = Guying(balance_engine)
    return guying


def test_guying_invalid_support_index(guying_basic_setup: Guying):
    guying = guying_basic_setup

    with pytest.raises(ValueError):
        guying.compute(
            index=10,  # out of range
            side='left',
            with_pulley=False,
            altitude=0,
            horizontal_distance=50,
        )

    with pytest.raises(AttributeError):
        # guying_side not in the authorized list
        guying.compute(
            index=0,
            side='xxx',  # type: ignore[arg-type]
            with_pulley=False,
            altitude=0,
            horizontal_distance=50,
        )

    with pytest.raises(TypeError):
        # with_pulley should be bool
        guying.compute(
            index=1,
            side='left',
            with_pulley=0,  # type: ignore[arg-type]
            altitude=0,
            horizontal_distance=50,
        )

    with pytest.raises(TypeError):
        # guying_altitude should be Real
        guying.compute(
            index=1,
            side='left',
            with_pulley=False,
            altitude="e",  # type: ignore[arg-type]
            horizontal_distance=50,
        )

    with pytest.raises(TypeError):
        # guying_horizontal_distance should be Real
        guying.compute(
            index=1,
            side='left',
            with_pulley=False,
            altitude=0,
            horizontal_distance="e",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError):
        # view should be in the authorized list
        guying.compute(
            index=1,
            side='left',
            with_pulley=False,
            altitude=0,
            horizontal_distance=50,
            view='xxx',  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError):
        # view should be str
        guying.compute(
            index=1,
            side='left',
            with_pulley=False,
            altitude=0,
            horizontal_distance=50,
            view=0,  # type: ignore[arg-type]
        )


def test_guying_support_index_with_pulley(guying_basic_setup: Guying):
    guying = guying_basic_setup

    # passing without pulley on first support
    guying.compute(
        index=0,
        side='left',
        with_pulley=False,
        altitude=0,
        horizontal_distance=50,
    )

    # should raise error with pulley on first support
    with pytest.raises(ValueError):
        guying.compute(
            index=0,  # out of range # type: ignore[arg-type]
            side='left',
            with_pulley=True,
            altitude=0,
            horizontal_distance=50,
        )

    # passing without pulley on last support
    guying.compute(
        index=3,
        side='left',
        with_pulley=False,
        altitude=0,
        horizontal_distance=50,
    )

    # should raise error with pulley on last support
    with pytest.raises(ValueError):
        guying.compute(
            index=3,  # out of range # type: ignore[arg-type]
            side='left',
            with_pulley=True,
            altitude=0,
            horizontal_distance=50,
        )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_span_view(guying_basic_setup: Guying):
    guying = guying_basic_setup
    expected_result_0 = guying.compute(
        index=1,
        side='left',
        with_pulley=False,
        altitude=0,
        horizontal_distance=50,
    )
    result_span_view_0 = guying.compute(
        index=1,
        side='right',
        with_pulley=False,
        altitude=0,
        horizontal_distance=50,
        view='span',
    )
    assert expected_result_0 == result_span_view_0

    expected_result_1 = guying.compute(
        index=1,
        side='right',
        with_pulley=False,
        altitude=0,
        horizontal_distance=50,
    )
    result_span_view_1 = guying.compute(
        index=2,
        side='left',
        with_pulley=False,
        altitude=0,
        horizontal_distance=50,
        view='span',
    )
    assert expected_result_1 == result_span_view_1


def test_span_view_warnings(guying_basic_setup: Guying, caplog):
    guying = guying_basic_setup

    with pytest.warns(UserWarning):
        guying.compute(
            index=1,
            side='right',
            with_pulley=False,
            altitude=0,
            horizontal_distance=50,
            view='span',
        )
