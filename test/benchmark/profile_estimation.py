# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Profile the state-management cost of distance-based estimations."""

import cProfile
import logging
import pstats
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mechaphlowers.api.section_study import SectionStudy
from mechaphlowers.core.models.estimation import BrentMethod, EstimationEngine
from mechaphlowers.data.catalog.catalog import sample_cable_catalog
from mechaphlowers.data.units import convert_weight_to_mass
from mechaphlowers.entities.arrays import SectionArray

logging.basicConfig(level=logging.WARNING)

OBSTACLE_POINT = np.array([250.0, 10.0, 20.0])
SUPPORT_MANIPULATION = {1: {"z": 1.0}}

PROFILE_TARGETS = [
    ("BalanceEngine.solve_change_state", "engine.py", "solve_change_state"),
    ("SectionStudy.solve_change_state", "section_study.py", "solve_change_state"),
    ("SectionStudy.save_state", "section_study.py", "save_state"),
    ("SectionStudy.restore_state", "section_study.py", "restore_state"),
    ("BalanceEngineCaretaker.save", "memento.py", "save"),
    ("BalanceEngineCaretaker.restore", "memento.py", "restore"),
    ("PositionEngine.update", "position_engine.py", "update"),
    ("PositionEngine.reset", "position_engine.py", "reset"),
    ("CoordsCalculator.reset", "points.py", "reset"),
    ("SectionStudy._solve_intermediate", "section_study.py", "_solve_intermediate"),
    ("SectionStudy.set_loads", "section_study.py", "set_loads"),
    ("SectionStudy._build_span_loads", "section_study.py", "_build_span_loads"),
    ("BalanceEngine.reset", "engine.py", "reset"),
    ("PositionEngine.point_distance", "position_engine.py", "point_distance"),
]

SCENARIOS = [
    {
        "name": "estimate_temperature",
        "known_value": 60.0,
        "bounds": (0.0, 120.0),
        "estimate_kwargs": {},
        "reference_kwargs": {"new_temperature": 60.0},
    },
    {
        "name": "estimate_wind",
        "known_value": 400.0,
        "bounds": (0.0, 1000.0),
        "estimate_kwargs": {},
        "reference_kwargs": {"wind_pressure": 400.0},
    },
    {
        "name": "estimate_load",
        "known_value": 50.0,
        "bounds": (0.0, 100.0),
        "estimate_kwargs": {"load_position_distance": 100.0},
        "reference_kwargs": {"load_position_distance": 100.0, "load_mass": 50.0},
    },
    {
        "name": "estimate_temperature_manipulated",
        "known_value": 60.0,
        "bounds": (0.0, 120.0),
        "estimate_kwargs": {},
        "reference_kwargs": {"new_temperature": 60.0},
        "support_manipulation": SUPPORT_MANIPULATION,
    },
]


def setup() -> SectionStudy:
    """Build and adjust a study equivalent to balance_engine_base_test."""
    cable_array = sample_cable_catalog.get_as_object(["ASTER600"])
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [50, 100, 50, 50],
                "crossarm_length": [10, 10, 10, 10],
                "line_angle": [0, 0, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 500, 500, np.nan],
                "insulator_mass": convert_weight_to_mass(
                    [1000.0, 500.0, 500.0, 1000.0]
                ),
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    study = SectionStudy(cable_array, section_array)
    study.solve_adjustment()
    return study


def apply_scenario_setup(study: SectionStudy, scenario: dict) -> None:
    support_manipulation = scenario.get("support_manipulation")
    if support_manipulation:
        study.modify_support(support_manipulation)


def get_reference_distance(scenario: dict) -> float:
    """Solve a fresh study at the known parameter and measure its distance."""
    study = setup()
    apply_scenario_setup(study, scenario)
    reference_kwargs = scenario["reference_kwargs"]

    if scenario["name"] == "estimate_load":
        n_spans = len(study.balance_engine.section_array.data.span_length) - 1
        load_positions = np.zeros(n_spans)
        load_masses = np.zeros(n_spans)
        load_positions[0] = reference_kwargs["load_position_distance"]
        load_masses[0] = reference_kwargs["load_mass"]
        study.set_loads(load_positions, load_masses)
    else:
        study.solve_change_state(**reference_kwargs)

    return study.position_engine.point_distance(
        span_index=0, point=OBSTACLE_POINT
    ).distance_3d


def estimate(study: SectionStudy, scenario: dict, target_distance: float):
    engine = EstimationEngine(study, method=BrentMethod(tol=0.1))
    kwargs = {
        "span_index": 0,
        "obstacle_point": OBSTACLE_POINT,
        "target_distance": target_distance,
        "bounds": scenario["bounds"],
        **scenario["estimate_kwargs"],
    }
    return getattr(engine, scenario["name"].split("_manipulated")[0])(**kwargs)


def profile_entry(stats: pstats.Stats, filename: str, function: str):
    entries = [
        values
        for (path, _line, name), values in stats.stats.items()
        if Path(path).name == filename and name == function
    ]
    calls = sum(values[1] for values in entries)
    cumulative = sum(values[3] for values in entries)
    return calls, cumulative


def run_scenario(scenario: dict) -> None:
    target_distance = get_reference_distance(scenario)
    timings = []
    for _ in range(3):
        study = setup()
        apply_scenario_setup(study, scenario)
        start = time.perf_counter()
        estimate(study, scenario, target_distance)
        timings.append(time.perf_counter() - start)

    profile_study = setup()
    apply_scenario_setup(profile_study, scenario)
    profiler = cProfile.Profile()
    profiler.enable()
    profiled_result = estimate(profile_study, scenario, target_distance)
    profiler.disable()
    stats = pstats.Stats(profiler)

    estimate_name = scenario["name"].split("_manipulated")[0]
    _, total_cumulative = profile_entry(
        stats, "engine.py", estimate_name
    )
    evaluation_count, _ = profile_entry(stats, "engine.py", "objective")

    print(f"\n=== {scenario['name']} ===")
    print(f"estimate_cum_ms={total_cumulative * 1000:.3f}")
    print(
        f"wall_min_ms={min(timings) * 1000:.3f}  evaluations={evaluation_count}  "
        f"target={scenario['known_value']:.6g}  value={profiled_result.value:.6g}  "
        f"converged={profiled_result.converged}  "
        f"residual={profiled_result.residual:.6g}"
    )
    print(f"{'Function':37} {'Calls':>6} {'Cum ms':>9} {'% est.':>8}")
    for label, filename, function in PROFILE_TARGETS:
        calls, cumulative = profile_entry(stats, filename, function)
        percent = cumulative / total_cumulative * 100 if total_cumulative else 0.0
        print(f"{label:37} {calls:6d} {cumulative * 1000:9.3f} {percent:7.2f}%")

    save_cumulative = profile_entry(stats, "section_study.py", "save_state")[1]
    restore_cumulative = profile_entry(stats, "section_study.py", "restore_state")[1]
    overhead_share = (
        (save_cumulative + restore_cumulative) / total_cumulative * 100
        if total_cumulative
        else 0.0
    )
    print(f"save_state + restore_state overhead share: {overhead_share:.2f}%")
    if estimate_name == "estimate_load":
        for function in ("_solve_intermediate", "set_loads"):
            cumulative = profile_entry(stats, "section_study.py", function)[1]
            share = cumulative / total_cumulative * 100 if total_cumulative else 0.0
            print(f"{function} share: {share:.2f}%")

    if scenario is SCENARIOS[0]:
        print("\nTop 15 functions by cumulative time:")
        top_entries = sorted(
            (
                (values[3], values[1], key)
                for key, values in stats.stats.items()
            ),
            reverse=True,
        )[:15]
        print(f"{'Function':42} {'Calls':>6} {'Cum ms':>9} {'% est.':>8}")
        for cumulative, calls, (path, line, function) in top_entries:
            label = f"{Path(path).stem}.{function}"
            percent = cumulative / total_cumulative * 100 if total_cumulative else 0.0
            print(f"{label:42} {calls:6d} {cumulative * 1000:9.3f} {percent:7.2f}%")


def main() -> None:
    for scenario in SCENARIOS:
        run_scenario(scenario)


if __name__ == "__main__":
    main()