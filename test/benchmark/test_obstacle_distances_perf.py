# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import math
import time
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.api.section_study import SectionStudy
from mechaphlowers.core.geometry.distances import DistanceResult
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.entities.errors import NoIntersectionPlaneWarning

N_SPANS = 10
SPAN_LENGTHS = [
    300.0,
    350.0,
    400.0,
    450.0,
    500.0,
    320.0,
    380.0,
    420.0,
    360.0,
    440.0,
]
N_ITERATIONS = 20


def _make_10span_study(cable_array: CableArray) -> SectionStudy:
    """10-span line (11 supports) with spans of varying length."""
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": [str(i) for i in range(1, 12)],
                "suspension": [False] + [True] * 9 + [False],
                "conductor_attachment_altitude": [
                    30.0,
                    45.0,
                    55.0,
                    60.0,
                    50.0,
                    65.0,
                    40.0,
                    35.0,
                    48.0,
                    57.0,
                    62.0,
                ],
                "crossarm_length": [0.0] + [5.0] * 9 + [0.0],
                "line_angle": [0.0] * 11,
                "insulator_length": [3.0] * 11,
                "span_length": [*SPAN_LENGTHS, np.nan],
                "insulator_mass": [1000.0] + [500.0] * 9 + [1000.0],
                "load_mass": [0.0] * 11,
                "load_position": [0.0] * 11,
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    return SectionStudy(cable_array=cable_array, section_array=section_array)


def _single_point_coords(span_index: int) -> np.ndarray:
    """One point at mid-span, below the cable."""
    length = SPAN_LENGTHS[span_index]
    return np.array([[0.5 * length, 0.0, 5.0]])


def _multipoint_coords(span_index: int, n_points: int) -> np.ndarray:
    """Polyline across the span: x from 0.1L to 0.9L, y alternating +-20, z in 0-15."""
    length = SPAN_LENGTHS[span_index]
    x = np.linspace(0.1 * length, 0.9 * length, n_points)
    y = np.where(np.arange(n_points) % 2 == 0, 20.0, -20.0)
    z = 2.0 + (np.arange(n_points) % 3) * 4.0
    return np.column_stack([x, y, z])


def _extract3(dr: DistanceResult) -> tuple[float, float, float]:
    return (
        dr.distance_3d,
        dr.distance_projection_u,
        dr.distance_projection_v,
    )


def _flatten_standard(
    result: dict[str, dict[int, DistanceResult]],
) -> list[DistanceResult]:
    return [dr for per_point in result.values() for dr in per_point.values()]


def _time_ms(run: Callable[[], object]) -> tuple[object, float]:
    """Warm up once, then return (last result, mean time per call in ms)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoIntersectionPlaneWarning)
        run()
        t0 = time.perf_counter()
        result = None
        for _ in range(N_ITERATIONS):
            result = run()
        elapsed_ms = (time.perf_counter() - t0) / N_ITERATIONS * 1000
    return result, elapsed_ms


def _print_table(rows: list[tuple[str, int, int, float]], title: str) -> None:
    col_w = [28, 14, 10, 14, 24]
    header = (
        f"{'Case':<{col_w[0]}}"
        f"{'#obstacles':>{col_w[1]}}"
        f"{'#points':>{col_w[2]}}"
        f"{'time (ms)':>{col_w[3]}}"
        f"{'time/obstacle (ms)':>{col_w[4]}}"
    )
    print(f"\n{title}")
    print(header)
    print("-" * sum(col_w))
    for case, n_obs, n_pts, time_ms in rows:
        per_obs = time_ms / n_obs
        print(
            f"{case:<{col_w[0]}}"
            f"{n_obs:>{col_w[1]}}"
            f"{n_pts:>{col_w[2]}}"
            f"{time_ms:>{col_w[3]}.3f}"
            f"{per_obs:>{col_w[4]}.3f}"
        )


def _assert_finite_positive(triples: list[tuple[float, float, float]]) -> None:
    for d3d, du, dv in triples:
        assert all(math.isfinite(v) for v in (d3d, du, dv))
        assert d3d > 0


@pytest.mark.benchmark
def test_perf_single_point_obstacles(cable_array_AM600: CableArray) -> None:
    """Group A: single-point obstacles, point_distance and standard per-point distances.

    Prints a timing table; does not assert on durations (benchmark only).
    """
    rows: list[tuple[str, int, int, float]] = []

    # ── A1: 1 obstacle, 1 point, point_distance on span 0 ───────────────────
    study = _make_10span_study(cable_array_AM600)
    study.solve_adjustment()
    pe = study.position_engine
    pe.add_obstacle("obs_0", 0, _single_point_coords(0))
    abs_point = pe.obstacles_dict()["obs_0"][0]

    def run_a1() -> list[tuple[float, float, float]]:
        return [_extract3(pe.point_distance(0, abs_point))]

    a1_result, a1_ms = _time_ms(run_a1)
    assert isinstance(a1_result, list) and len(a1_result) == 1
    _assert_finite_positive(a1_result)
    rows.append(("A1 point_distance", 1, 1, a1_ms))

    # ── A2: 1 single-point obstacle per span (10 obstacles) ─────────────────
    study = _make_10span_study(cable_array_AM600)
    study.solve_adjustment()
    pe = study.position_engine
    for span in range(N_SPANS):
        pe.add_obstacle(f"obs_{span}", span, _single_point_coords(span))

    def run_a2() -> list[tuple[float, float, float]]:
        return [
            _extract3(dr)
            for dr in _flatten_standard(pe.get_distances_from_obstacles())
        ]

    a2_result, a2_ms = _time_ms(run_a2)
    assert len(a2_result) == 10
    _assert_finite_positive(a2_result)
    rows.append(("A2 1 pt/obstacle", 10, 10, a2_ms))

    # ── A3: 3 single-point obstacles per span (30 obstacles) ────────────────
    study = _make_10span_study(cable_array_AM600)
    study.solve_adjustment()
    pe = study.position_engine
    for span in range(N_SPANS):
        for k in range(3):
            coords = _single_point_coords(span)
            coords[0, 0] += (k - 1) * 10.0
            coords[0, 1] += (k - 1) * 5.0
            pe.add_obstacle(f"obs_{span}_{k}", span, coords)

    def run_a3() -> list[tuple[float, float, float]]:
        return [
            _extract3(dr)
            for dr in _flatten_standard(pe.get_distances_from_obstacles())
        ]

    a3_result, a3_ms = _time_ms(run_a3)
    assert len(a3_result) == 30
    _assert_finite_positive(a3_result)
    rows.append(("A3 3x 1 pt/obstacle", 30, 30, a3_ms))

    _print_table(rows, "Group A: single-point obstacles")


@pytest.mark.benchmark
def test_perf_multipoint_obstacles(cable_array_AM600: CableArray) -> None:
    """Group B: multipoint obstacles (1 per span, 10 obstacles).

    Prints a timing table; does not assert on durations (benchmark only).
    """
    rows: list[tuple[str, int, int, float]] = []

    # ── B1: 3 points per obstacle, standard per-point distances ─────────────
    study = _make_10span_study(cable_array_AM600)
    study.solve_adjustment()
    pe = study.position_engine
    for span in range(N_SPANS):
        pe.add_obstacle(f"obs_{span}", span, _multipoint_coords(span, 3))

    def run_b1() -> list[tuple[float, float, float]]:
        return [
            _extract3(dr)
            for dr in _flatten_standard(pe.get_distances_from_obstacles())
        ]

    b1_result, b1_ms = _time_ms(run_b1)
    assert len(b1_result) == 30
    _assert_finite_positive(b1_result)
    rows.append(("B1 per-point, 3 pts", 10, 30, b1_ms))

    # ── B2: 10 points per obstacle, new minimum vs per-point ────────────────
    study = _make_10span_study(cable_array_AM600)
    study.solve_adjustment()
    pe = study.position_engine
    for span in range(N_SPANS):
        pe.add_obstacle(f"obs_{span}", span, _multipoint_coords(span, 10))

    def run_b2_min() -> list[tuple[float, float, float] | None]:
        results = pe.get_segment_distances_from_obstacles()
        return [
            None if res is None else _extract3(res.distance_result)
            for res in results.values()
        ]

    def run_b2_per_point() -> list[tuple[float, float, float]]:
        return [
            _extract3(dr)
            for dr in _flatten_standard(pe.get_distances_from_obstacles())
        ]

    b2_min_result, b2_min_ms = _time_ms(run_b2_min)
    valid_min = [t for t in b2_min_result if t is not None]
    assert len(valid_min) == 10
    _assert_finite_positive(valid_min)
    rows.append(("B2 new min, 10 pts", 10, 100, b2_min_ms))

    b2_pp_result, b2_pp_ms = _time_ms(run_b2_per_point)
    assert len(b2_pp_result) == 100
    _assert_finite_positive(b2_pp_result)
    rows.append(("B2 per-point, 10 pts", 10, 100, b2_pp_ms))

    # Minimum over the polyline must not exceed the minimum over its points.
    min_results = pe.get_segment_distances_from_obstacles()
    per_point = pe.get_distances_from_obstacles()
    for name, res in min_results.items():
        assert res is not None
        point_min = min(dr.distance_3d for dr in per_point[name].values())
        assert res.distance_result.distance_3d <= point_min + 1e-9

    _print_table(rows, "Group B: multipoint obstacles")
