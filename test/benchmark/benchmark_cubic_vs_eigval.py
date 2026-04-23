# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Performance comparison: Cardano vs EigvalBatch cubic solvers.

Uses the same balance-engine context as profile_balance.py
(4 supports with loads, wind+ice+temperature).
"""

import timeit

import numpy as np
import pandas as pd

from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.data.catalog.catalog import sample_cable_catalog
from mechaphlowers.entities.arrays import SectionArray


def setup():
    cable = sample_cable_catalog.get_as_object(['ASTER600'])
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
                "insulator_mass": [1000, 500, 500, 1000],
                "load_mass": [500, 1000, 500, np.nan],
                "load_position": [0.2, 0.4, 0.6, np.nan],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})

    engine = BalanceEngine(cable_array=cable, section_array=section_array)
    engine.solve_adjustment()
    return engine


def run_change_state(engine):
    engine.solve_change_state(
        wind_pressure=np.array([-200, -200, -200, -200]),
        ice_thickness=np.array([0.02, 0.02, 0.02, 0.02]),
        new_temperature=np.array([5, 5, 5, 5]),
    )


def bench(label, N=100, repeats=5):
    engine = setup()
    run_change_state(engine)  # warmup
    times = timeit.repeat(
        lambda: run_change_state(engine), number=N, repeat=repeats
    )
    avg = min(times) / N * 1000
    print(f"  {label:25s}: {avg:.3f} ms/call")
    return avg


if __name__ == "__main__":
    # We must reload model_ducloux to pick up the new config each time.
    # Since _cubic_solver is a module-level variable set at import time,
    # we need to patch it directly.
    import mechaphlowers.core.models.balance.models.model_ducloux as md
    from mechaphlowers.numeric.analytical_cubic import AnalyticalRealSolver
    from mechaphlowers.numeric.cubic import CardanoSolver
    from mechaphlowers.numeric.eigval_batch_lapack import EigvalBatchSolver

    results = {}

    print("=== Full integration: solve_change_state ===\n")

    # Eigval batch (default)
    md._cubic_solver = EigvalBatchSolver()
    results["eigval_batch (LAPACK)"] = bench("eigval_batch (LAPACK)")

    # Cardano
    md._cubic_solver = CardanoSolver()
    results["cardano (numpy)"] = bench("cardano (numpy)")

    # Analytical real (trigonometric/Cardano, no LAPACK)
    md._cubic_solver = AnalyticalRealSolver()
    results["analytical_real"] = bench("analytical_real")

    # Restore default
    md._cubic_solver = md._build_cubic_solver()

    print("\n=== Summary ===")
    eigval_ms = results["eigval_batch (LAPACK)"]
    cardano_ms = results["cardano (numpy)"]
    analytical_ms = results["analytical_real"]

    print(f"\n  cardano        : {cardano_ms:.3f} ms  (baseline)")
    print(
        f"  eigval_batch   : {eigval_ms:.3f} ms  ({(cardano_ms - eigval_ms) / cardano_ms * 100:.1f}% faster than cardano)"
    )
    print(
        f"  analytical_real: {analytical_ms:.3f} ms  ({(cardano_ms - analytical_ms) / cardano_ms * 100:.1f}% faster than cardano)"
    )

    best_ms = min(eigval_ms, analytical_ms)
    best_label = (
        "eigval_batch" if eigval_ms <= analytical_ms else "analytical_real"
    )
    print(f"\n  Best solver: {best_label} ({best_ms:.3f} ms)")
    if analytical_ms < eigval_ms:
        print(
            f"  analytical_real is {(eigval_ms - analytical_ms) / eigval_ms * 100:.1f}% faster than eigval_batch"
        )
    else:
        print(
            f"  eigval_batch is {(analytical_ms - eigval_ms) / analytical_ms * 100:.1f}% faster than analytical_real"
        )
