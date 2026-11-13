# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from mechaphlowers.config import options
from mechaphlowers.core.models.balance.engine import BalanceEngine
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


def jacobians_banded_and_dense(model, solver, monkeypatch):
    rng = np.random.default_rng(0)
    # move away from equilibrium to test a generic point
    model.state_vector = model.state_vector + rng.uniform(
        -0.05, 0.05, model.state_vector.shape
    )
    model.update()
    objective = model.objective_function()

    jacobian_banded = solver.jacobian(objective, model, solver.perturb)
    monkeypatch.setattr(
        type(model), "jacobian_half_bandwidth", property(lambda self: None)
    )
    jacobian_dense = solver.jacobian(objective, model, solver.perturb)
    return jacobian_banded, jacobian_dense


def solve_change_state(engine):
    engine.solve_change_state(
        wind_pressure=np.full(8, -200.0),
        ice_thickness=np.full(8, 0.02),
        new_temperature=np.full(8, 5.0),
    )


def test_banded_jacobian_matches_dense__adjustment(
    balance_engine_long, monkeypatch
):
    engine = balance_engine_long
    engine.balance_model.adjustment = True
    banded, dense = jacobians_banded_and_dense(
        engine.balance_model, engine.solver_adjustment, monkeypatch
    )
    assert_allclose(banded, dense, rtol=0, atol=1e-6 * np.abs(dense).max())


def test_banded_jacobian_matches_dense__change_state_with_loads(
    balance_engine_long, monkeypatch
):
    engine = balance_engine_long
    solve_change_state(engine)
    assert engine.balance_model.nodes.has_load_on_span.any()
    banded, dense = jacobians_banded_and_dense(
        engine.balance_model, engine.solver_change_state, monkeypatch
    )
    assert_allclose(banded, dense, rtol=0, atol=1e-6 * np.abs(dense).max())


def test_banded_jacobian_matches_dense__load_model(
    balance_engine_long, monkeypatch
):
    engine = balance_engine_long
    solve_change_state(engine)
    load_model = engine.balance_model.load_model
    assert len(load_model.state_vector) > 3  # banded path is used
    banded, dense = jacobians_banded_and_dense(
        load_model, engine.balance_model.load_solver, monkeypatch
    )
    assert_allclose(banded, dense, rtol=0, atol=1e-6 * np.abs(dense).max())


def test_banded_jacobian_inactive_uses_dense(balance_engine_long, monkeypatch):
    engine = balance_engine_long
    engine.balance_model.adjustment = True
    model = engine.balance_model
    solver = engine.solver_adjustment
    objective = model.objective_function()

    called = []
    monkeypatch.setattr(
        solver, "_banded_jacobian", lambda *a, **k: called.append(True)
    )
    monkeypatch.setattr(
        options.solver.acceleration.banded_jacobian, "active", False
    )
    solver.jacobian(objective, model, solver.perturb)
    assert not called

    monkeypatch.setattr(
        options.solver.acceleration.banded_jacobian, "active", True
    )
    solver.jacobian(objective, model, solver.perturb)
    assert called


def test_dense_jacobian_called_when_banded_inactive(
    balance_engine_long, monkeypatch
):
    engine = balance_engine_long
    engine.balance_model.adjustment = True
    model = engine.balance_model
    solver = engine.solver_adjustment
    objective = model.objective_function()
    n = len(objective)
    assert 2 * model.jacobian_half_bandwidth + 1 < n  # banded would be used

    monkeypatch.setattr(
        options.solver.acceleration.banded_jacobian, "active", False
    )
    banded_spy = MagicMock(side_effect=solver._banded_jacobian)
    monkeypatch.setattr(solver, "_banded_jacobian", banded_spy)
    delta_d_spy = MagicMock(side_effect=solver._delta_d)
    monkeypatch.setattr(solver, "_delta_d", delta_d_spy)

    jacobian = solver.jacobian(objective, model, solver.perturb)

    banded_spy.assert_not_called()
    # dense jacobian perturbs one variable at a time
    assert delta_d_spy.call_count == n
    assert jacobian.shape == (n, n)


def test_banded_jacobian_band_size_from_config(
    balance_engine_long, monkeypatch
):
    model = balance_engine_long.balance_model
    banded_config = options.solver.acceleration.banded_jacobian
    monkeypatch.setattr(banded_config, "node_band_size", 5)
    monkeypatch.setattr(banded_config, "load_band_size", 2)
    assert model.jacobian_half_bandwidth == 5
    assert model.load_model.jacobian_half_bandwidth == 2
