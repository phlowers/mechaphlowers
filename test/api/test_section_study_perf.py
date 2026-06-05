# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.api.section_study import SectionStudy
from mechaphlowers.entities.arrays import CableArray, SectionArray

if TYPE_CHECKING:
    pass


def _make_8support_study(cable_array: CableArray) -> SectionStudy:
    """8-support line with spans of varying length."""
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4", "5", "6", "7", "8"],
                "suspension": [
                    False,
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
                    35.0,
                ],
                "crossarm_length": [0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0],
                "line_angle": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "insulator_length": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                "span_length": [
                    400.0,
                    350.0,
                    450.0,
                    300.0,
                    500.0,
                    380.0,
                    420.0,
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
                    1000.0,
                ],
                "load_mass": [0.0] * 8,
                "load_position": [0.0] * 8,
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    return SectionStudy(cable_array=cable_array, section_array=section_array)


def _make_12support_study(cable_array: CableArray) -> SectionStudy:
    """12-support plain line for size-scaling comparison."""
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": [
                    "1",
                    "2",
                    "3",
                    "4",
                    "5",
                    "6",
                    "7",
                    "8",
                    "9",
                    "10",
                    "11",
                    "12",
                ],
                "suspension": [
                    False,
                    True,
                    True,
                    True,
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
                    35.0,
                    50.0,
                    58.0,
                    42.0,
                    38.0,
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
                    5.0,
                    5.0,
                    5.0,
                    0.0,
                ],
                "line_angle": [0.0] * 12,
                "insulator_length": [3.0] * 12,
                "span_length": [
                    400.0,
                    350.0,
                    450.0,
                    300.0,
                    500.0,
                    380.0,
                    420.0,
                    370.0,
                    410.0,
                    340.0,
                    460.0,
                    np.nan,
                ],
                "insulator_mass": [1000.0] + [500.0] * 10 + [1000.0],
                "load_mass": [0.0] * 12,
                "load_position": [0.0] * 12,
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    return SectionStudy(cable_array=cable_array, section_array=section_array)


@pytest.mark.benchmark
def test_perf_data_and_change_state_baseline_vs_manipulations(
    cable_array_AM600: CableArray,
) -> None:
    """Compare .data and solve_change_state timing between:
    - a plain 8-support line (baseline),
    - the same 8-support line with 4 support manipulations, 1 rope manipulation
      and 4 virtual supports,
    - a plain 12-support line (size-scaling reference).

    Prints a timing table; does not assert on durations (benchmark only).
    """
    n_iterations = 20

    def _measure(study: SectionStudy) -> tuple[float, float]:
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            _ = study.balance_engine.section_array.data
        data_s = (time.perf_counter() - t0) / n_iterations

        t0 = time.perf_counter()
        for _ in range(n_iterations):
            study.solve_change_state(new_temperature=15.0)
        change_state_s = (time.perf_counter() - t0) / n_iterations

        return data_s, change_state_s

    # ── baseline: plain 8-support ────────────────────────────────────────────
    study_base = _make_8support_study(cable_array_AM600)
    study_base.solve_adjustment()
    baseline_data_s, baseline_change_state_s = _measure(study_base)

    # ── 8-support with manipulations ─────────────────────────────────────────
    study_manip = _make_8support_study(cable_array_AM600)
    # 4 support manipulations (supports 1, 2, 4, 5)
    study_manip.manipulation.shift_support(
        {
            1: {"z": 1.0},
            2: {"z": -1.0, "y": 0.5},
            4: {"z": 2.0},
            5: {"y": -0.5},
        }
    )
    # 1 rope manipulation (support 3)
    study_manip.manipulation.add_rope({3: 4.5})
    study_manip.solve_adjustment()
    manip_data_s, manip_change_state_s = _measure(study_manip)

    # ── size-scaling reference: plain 12-support ──────────────────────────────
    study_12 = _make_12support_study(cable_array_AM600)
    study_12.solve_adjustment()
    ref12_data_s, ref12_change_state_s = _measure(study_12)

    # ── report ────────────────────────────────────────────────────────────────
    col_w = [30, 16, 24, 18, 8]
    header = (
        f"{'Measurement':<{col_w[0]}}"
        f"{'8-support (ms)':>{col_w[1]}}"
        f"{'8-support+manip (ms)':>{col_w[2]}}"
        f"{'12-support (ms)':>{col_w[3]}}"
        f"{'manip ratio':>{col_w[4]}}"
    )
    print(f"\n{header}")
    print("-" * sum(col_w))
    for label, base, manip, ref12 in (
        (".data", baseline_data_s, manip_data_s, ref12_data_s),
        (
            "solve_change_state",
            baseline_change_state_s,
            manip_change_state_s,
            ref12_change_state_s,
        ),
    ):
        ratio = manip / ref12 if ref12 > 0 else float("inf")
        print(
            f"{label:<{col_w[0]}}"
            f"{base * 1000:>{col_w[1]}.3f}"
            f"{manip * 1000:>{col_w[2]}.3f}"
            f"{ref12 * 1000:>{col_w[3]}.3f}"
            f"{ratio:>{col_w[4]}.2f}x"
        )
    print(
        "expected: solve_change_state overhead from manipulations should be "
        "comparable to the plain size increase from 8 to 12 supports"
    )
