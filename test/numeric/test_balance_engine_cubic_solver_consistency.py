# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd
import pytest

import mechaphlowers.core.models.balance.models.model_ducloux as model_ducloux
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.data.units import convert_weight_to_mass
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.numeric.analytical_cubic import AnalyticalRealSolver
from mechaphlowers.numeric.cubic import CardanoSolver
from mechaphlowers.numeric.eigval_batch_lapack import EigvalBatchSolver

SOLVER_CLASSES = (
    EigvalBatchSolver,
    CardanoSolver,
    AnalyticalRealSolver,
)


def _run_balance_engine_with_solver(
    solver_cls: type,
    cable_array_AM600: CableArray,
    monkeypatch: pytest.MonkeyPatch,
) -> BalanceEngine:
    monkeypatch.setattr(model_ducloux, "_cubic_solver", solver_cls())

    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": [0, 10, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": convert_weight_to_mass(
                    [1000, 500, 500, 1000]
                ),
                "load_mass": convert_weight_to_mass([0, 1000, 0, np.nan]),
                "load_position": [0, 0.4, 0, np.nan],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})

    engine = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array,
    )
    engine.solve_adjustment()
    engine.solve_change_state()
    return engine


def _extract_engine_results(engine: BalanceEngine) -> dict[str, np.ndarray]:
    return {
        "displacement": engine.balance_model.chain_displacement().copy(),
        "L_ref": engine.balance_model.L_ref.copy(),
        "vhl_under_chain": engine.balance_model.vhl_under_chain()
        .vhl_matrix.value()
        .copy(),
    }


@pytest.mark.integration
def test_balance_engine_load_one_span_results_match_across_cubic_solvers(
    cable_array_AM600: CableArray,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_results: dict[str, np.ndarray] | None = None

    for solver_cls in SOLVER_CLASSES:
        engine = _run_balance_engine_with_solver(
            solver_cls, cable_array_AM600, monkeypatch
        )
        results = _extract_engine_results(engine)

        if reference_results is None:
            reference_results = results

        for key, expected in reference_results.items():
            np.testing.assert_allclose(results[key], expected, atol=1e-6)


@pytest.mark.integration
@pytest.mark.parametrize("solver_cls", SOLVER_CLASSES)
def test_balance_engine_load_one_span_matches_known_displacements(
    cable_array_AM600: CableArray,
    monkeypatch: pytest.MonkeyPatch,
    solver_cls: type,
) -> None:
    engine = _run_balance_engine_with_solver(
        solver_cls, cable_array_AM600, monkeypatch
    )

    expected_displacement = np.array(
        [
            [2.98631241983472, 0.0649786506874284, -0.278775727235584],
            [0.106794467626418, 0.888327055351789, -2.86347166642424],
            [-0.0276983016774005, 1.16170727466419, -2.7658035020725],
            [-2.9774999521832, -0.0657854850521439, -0.360785676968292],
        ]
    )

    np.testing.assert_allclose(
        engine.balance_model.chain_displacement(),
        expected_displacement,
        atol=1e-4,
    )
