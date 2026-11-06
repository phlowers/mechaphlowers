# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.core.models.balance.models.model_ducloux import (
    BalanceModel,
)
from mechaphlowers.data.catalog.catalog import sample_cable_catalog
from mechaphlowers.entities.arrays import SectionArray


@pytest.fixture
def balance_engine_long() -> BalanceEngine:
    """Section with anchors, line angles, crossarms and loads, long enough for
    the banded jacobian to be used (2 * 3 + 1 < 2 * nb_supports)."""
    attachment_altitude = [30, 50, 60, 45, 55, 40, 50, 65]
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": [str(i) for i in range(8)],
                "suspension": [False] + [True] * 6 + [False],
                "conductor_attachment_altitude": attachment_altitude,
                "crossarm_length": [0, 10, -10, 5, 0, -5, 10, 0],
                "line_angle": [0, 10, 0, 0, 15, 0, 5, 0],
                "insulator_length": [3] * 8,
                "span_length": [500, 300, 400, 450, 350, 380, 420, np.nan],
                "insulator_mass": [1000] + [500] * 6 + [1000],
            }
        )
    )
    section_array.add_units({"line_angle": "grad"})
    section_array.set_sagging_parameter(2000)
    section_array.set_sagging_temperature(15)

    cable = sample_cable_catalog.get_as_object(["ASTER600"])
    engine = BalanceEngine(cable_array=cable, section_array=section_array)
    engine.set_loads(
        load_position_distance=[100, 0, 160, 0, 0, 230, 0],
        load_mass=[500, 0, 1000, 0, 0, 800, 0],
    )
    engine.solve_adjustment()
    return engine


def jacobians_banded_and_dense(engine, solver, monkeypatch):
    model = engine.balance_model
    rng = np.random.default_rng(0)
    # move away from equilibrium to test a generic point
    model.state_vector = model.state_vector + rng.uniform(
        -0.05, 0.05, model.state_vector.shape
    )
    model.update()
    objective = model.objective_function()

    jacobian_banded = solver.jacobian(objective, model, solver.perturb)
    monkeypatch.setattr(
        BalanceModel, "jacobian_half_bandwidth", property(lambda self: None)
    )
    jacobian_dense = solver.jacobian(objective, model, solver.perturb)
    return jacobian_banded, jacobian_dense


def test_banded_jacobian_matches_dense__adjustment(
    balance_engine_long, monkeypatch
):
    engine = balance_engine_long
    engine.balance_model.adjustment = True
    banded, dense = jacobians_banded_and_dense(
        engine, engine.solver_adjustment, monkeypatch
    )
    assert_allclose(banded, dense, rtol=0, atol=1e-6 * np.abs(dense).max())


def test_banded_jacobian_matches_dense__change_state_with_loads(
    balance_engine_long, monkeypatch
):
    engine = balance_engine_long
    engine.solve_change_state(
        wind_pressure=np.full(8, -200.0),
        ice_thickness=np.full(8, 0.02),
        new_temperature=np.full(8, 5.0),
    )
    assert engine.balance_model.nodes.has_load_on_span.any()
    banded, dense = jacobians_banded_and_dense(
        engine, engine.solver_change_state, monkeypatch
    )
    assert_allclose(banded, dense, rtol=0, atol=1e-6 * np.abs(dense).max())
