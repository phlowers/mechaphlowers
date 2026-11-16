# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.core.models.balance.memento import BalanceEngineCaretaker
from mechaphlowers.entities.arrays import CableArray, SectionArray


def make_section_array() -> SectionArray:
    return SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [50, 100, 50, 50],
                "crossarm_length": [10, 10, 10, 10],
                "line_angle": [0, 0, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 500, 500, np.nan],
                "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )


@pytest.fixture
def narcisse_engine(cable_array_NARCISSE600G: CableArray) -> BalanceEngine:
    engine = BalanceEngine(
        cable_array=cable_array_NARCISSE600G,
        section_array=make_section_array(),
    )
    engine.solve_adjustment()
    return engine


def virgin_parameter(engine: BalanceEngine) -> np.ndarray:
    engine.solve_change_state(new_temperature=15)
    return engine.span_model.parameter.copy()


def test_history_only_changes_after_commit(
    narcisse_engine: BalanceEngine,
) -> None:
    engine = narcisse_engine
    reference = virgin_parameter(engine)
    engine.solve_change_state(new_temperature=15)
    np.testing.assert_allclose(engine.span_model.parameter, reference)

    engine.solve_change_state(ice_thickness=0.03, new_temperature=-5)
    engine.commit_deformation_state()
    engine.solve_change_state(new_temperature=15)
    after_commit = engine.span_model.parameter.copy()

    # permanent elongation: the cable is longer, so it sags more
    assert np.all(after_commit[:3] < reference[:3])
    state = engine.deformation_state
    assert state.max_strain.shape == (2, 4)
    assert np.all(state.max_strain >= 0)
    assert np.any(state.max_strain > 0)

    engine.reset_deformation_state()
    np.testing.assert_allclose(virgin_parameter(engine), reference)


def test_ratchet_commits_automatically(narcisse_engine: BalanceEngine) -> None:
    engine = narcisse_engine
    engine.solve_change_state(new_temperature=15)
    engine.ratchet = False
    engine.solve_change_state(ice_thickness=0.03, new_temperature=-5)
    assert np.all(engine.deformation_state.max_strain == 0)

    engine.ratchet = True
    engine.solve_change_state(ice_thickness=0.03, new_temperature=-5)
    assert np.any(engine.deformation_state.max_strain > 0)


def test_loads_keep_history_and_raise_loaded_span(
    narcisse_engine: BalanceEngine,
) -> None:
    engine = narcisse_engine
    engine.solve_change_state(new_temperature=15)
    state_before = engine.deformation_state

    engine.set_loads([250, 0, 0], [500, 0, 0])
    assert engine.deformation_state is state_before

    engine.solve_change_state(ice_thickness=0.03, new_temperature=-5)
    engine.commit_deformation_state()
    loaded = engine.deformation_state.max_strain

    engine.reset_deformation_state()
    engine.set_loads([0, 0, 0], [0, 0, 0])
    engine.solve_change_state(ice_thickness=0.03, new_temperature=-5)
    engine.commit_deformation_state()
    unloaded = engine.deformation_state.max_strain

    assert loaded[0, 0] > unloaded[0, 0]
    assert np.all(loaded >= unloaded)


def test_memento_restores_deformation_state(
    narcisse_engine: BalanceEngine,
) -> None:
    engine = narcisse_engine
    engine.solve_change_state(ice_thickness=0.03, new_temperature=-5)
    engine.commit_deformation_state()
    saved_state = engine.deformation_state
    caretaker = BalanceEngineCaretaker(engine)
    memento = caretaker.save()

    engine.reset_deformation_state()
    assert engine.deformation_state is not saved_state
    assert np.all(engine.deformation_state.max_strain == 0)

    caretaker.restore(memento)
    assert engine.deformation_state is saved_state


def test_linear_cable_commit_is_noop(
    balance_engine_base_test: BalanceEngine,
) -> None:
    engine = balance_engine_base_test
    engine.solve_adjustment()
    reference = virgin_parameter(engine)

    engine.solve_change_state(ice_thickness=0.03, new_temperature=-5)
    engine.commit_deformation_state()
    assert engine.deformation_state.max_strain is None

    np.testing.assert_allclose(virgin_parameter(engine), reference)
