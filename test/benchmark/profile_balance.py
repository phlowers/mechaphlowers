# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Profile solve_change_state to identify numpy hot spots."""

import cProfile
import io
import pstats
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
        )
    )
    section_array.add_units({"line_angle": "grad"})
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    engine = BalanceEngine(cable_array=cable, section_array=section_array)
    engine.solve_adjustment()
    return engine


def run_change_state(engine):
    engine.solve_change_state(
        wind_pressure=np.array([-200, -200, -200, -200]),
        ice_thickness=np.array([0.02, 0.02, 0.02, 0.02]),
        new_temperature=np.array([5, 5, 5, 5]),
    )


def profile_run():
    engine = setup()

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(50):
        run_change_state(engine)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    print("=== CUMULATIVE ===")
    ps.print_stats(60)
    print(s.getvalue())

    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2)
    ps2.sort_stats('tottime')
    print("\n=== TOTTIME (self time) ===")
    ps2.print_stats(60)
    print(s2.getvalue())


def benchmark():
    engine = setup()

    # Warmup
    run_change_state(engine)

    N = 100
    times = timeit.repeat(lambda: run_change_state(engine), number=N, repeat=5)
    avg_per_call = min(times) / N * 1000  # ms
    print(
        f"\nBenchmark: {avg_per_call:.3f} ms per solve_change_state call (best of 5 repeats, {N} calls each)"
    )
    return avg_per_call


if __name__ == "__main__":
    profile_run()
    benchmark()
