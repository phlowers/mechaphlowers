# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pytest import fixture

from mechaphlowers.core.models.balance.engine import (
    BalanceEngine,
)
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.entities.errors import (
    ConvergenceError,
)


@fixture
def balance_engine_simple(cable_array_AM600: CableArray) -> BalanceEngine:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 0, 0, 0],
                "line_angle": [0, 0, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [1000, 500, 500, 1000],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    return BalanceEngine(
        cable_array=cable_array_AM600, section_array=section_array
    )


@fixture
def section_array_arm() -> SectionArray:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": [0, 0, 0, 0],
                "insulator_length": [0, 3, 3, 0],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [1000, 500, 500, 1000],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    return section_array


def test_element_initialisation(balance_engine_simple: BalanceEngine):
    print("\n")
    print(balance_engine_simple.balance_model)
    repr_be = balance_engine_simple.__repr__()
    assert repr_be.startswith("BalanceEngine")
    assert balance_engine_simple.support_number == 4


def test_element_change_state(balance_engine_simple: BalanceEngine):
    with pytest.warns(UserWarning):
        balance_engine_simple.solve_change_state()

    balance_engine_simple.solve_adjustment()

    balance_engine_simple.solve_change_state()


def test_change_state_defaults(balance_engine_simple: BalanceEngine):
    span_shape = balance_engine_simple.section_array.data.span_length.shape
    balance_engine_simple.solve_adjustment()
    balance_engine_simple.solve_change_state(
        wind_pressure=None, ice_thickness=None, new_temperature=None
    )
    # Check that default values are used
    np.testing.assert_array_equal(
        balance_engine_simple.balance_model.cable_loads.wind_pressure,
        np.zeros(span_shape),
    )
    np.testing.assert_array_equal(
        balance_engine_simple.balance_model.cable_loads.ice_thickness,
        np.zeros(span_shape),
    )
    np.testing.assert_array_equal(
        balance_engine_simple.deformation_model.current_temperature,
        np.full(span_shape, 15.0),
    )


def test_change_state_logic(balance_engine_simple: BalanceEngine):
    span_shape = balance_engine_simple.section_array.data.span_length.shape
    balance_engine_simple.solve_adjustment()

    # Test with wind_pressure and ice_thickness provided
    balance_engine_simple.solve_change_state(
        wind_pressure=np.full(span_shape, 50.0),
        ice_thickness=1.0,
        new_temperature=None,
    )
    np.testing.assert_array_equal(
        balance_engine_simple.balance_model.cable_loads.wind_pressure,
        np.full(span_shape, 50.0),
    )
    np.testing.assert_array_equal(
        balance_engine_simple.balance_model.cable_loads.ice_thickness,
        np.zeros(span_shape) + 1.0,
    )
    np.testing.assert_array_equal(
        balance_engine_simple.deformation_model.current_temperature,
        np.full(span_shape, 15.0),
    )

    # Test with floats provided
    balance_engine_simple.solve_change_state(
        wind_pressure=100.0,
        ice_thickness=1.0,
        new_temperature=10.0,
    )
    np.testing.assert_array_equal(
        balance_engine_simple.balance_model.cable_loads.wind_pressure,
        np.full(span_shape, 100.0),
    )
    np.testing.assert_array_equal(
        balance_engine_simple.balance_model.cable_loads.ice_thickness,
        np.full(span_shape, 1.0),
    )
    np.testing.assert_array_equal(
        balance_engine_simple.deformation_model.current_temperature,
        np.full(span_shape, 10.0),
    )

    # Test with only ice_thickness provided
    balance_engine_simple.solve_change_state(
        wind_pressure=None,
        ice_thickness=np.full(span_shape, 0.01),
        new_temperature=None,
    )
    np.testing.assert_array_equal(
        balance_engine_simple.balance_model.cable_loads.wind_pressure,
        np.zeros(span_shape),
    )
    np.testing.assert_array_equal(
        balance_engine_simple.balance_model.cable_loads.ice_thickness,
        np.full(span_shape, 0.01),
    )
    np.testing.assert_array_equal(
        balance_engine_simple.deformation_model.current_temperature,
        np.full(span_shape, 15.0),
    )

    # Test with only new_temperature provided
    balance_engine_simple.solve_change_state(
        wind_pressure=None,
        ice_thickness=None,
        new_temperature=np.full(span_shape, 25.0),
    )
    np.testing.assert_array_equal(
        balance_engine_simple.balance_model.cable_loads.wind_pressure,
        np.zeros(span_shape),
    )
    np.testing.assert_array_equal(
        balance_engine_simple.balance_model.cable_loads.ice_thickness,
        np.zeros(span_shape),
    )
    np.testing.assert_array_equal(
        balance_engine_simple.deformation_model.current_temperature,
        np.full(span_shape, 25.0),
    )


def test_change_state__input_errors(balance_engine_simple: BalanceEngine):
    balance_engine_simple.solve_adjustment()

    with pytest.raises(ValueError):
        balance_engine_simple.solve_change_state(
            wind_pressure=np.array([1, 2]),  # Incorrect shape
            ice_thickness=None,
            new_temperature=None,
        )

    with pytest.raises(TypeError):
        balance_engine_simple.solve_change_state(
            wind_pressure=None,
            ice_thickness="0",  # type: ignore[arg-type]
            new_temperature=None,
        )


def test_load_span__node_span_coherence_with_balance_model(
    cable_array_AM600: CableArray,
):
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
                "insulator_mass": [100, 50, 50, 100],
                "load_mass": [0, 1000, 0, np.nan],
                "load_position": [0.2, 0.4, 0.6, np.nan],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})

    balance_engine_angles_arm = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array,
    )

    balance_engine_angles_arm.solve_adjustment()

    balance_engine_angles_arm.solve_change_state()
    span_model = balance_engine_angles_arm.balance_model.nodes_span_model
    assert len(span_model.span_length) == 5
    expected_right_span_length = balance_engine_angles_arm.balance_model.load_model.span_model_right.span_length  # type: ignore[attr-defined]
    expected_left_span_length = balance_engine_angles_arm.balance_model.load_model.span_model_left.span_length  # type: ignore[attr-defined]
    old_value = balance_engine_angles_arm.span_model.span_length

    assert span_model.span_length[1] == expected_left_span_length
    assert span_model.span_length[2] == expected_right_span_length
    assert span_model.span_length[3] == old_value[2]

    assert balance_engine_angles_arm.balance_model.has_loads


def test_load_span__check_node_span_changes(cable_array_AM600: CableArray):
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
                "insulator_mass": [100, 50, 50, 100],
                "load_mass": [0, 1000, 0, np.nan],
                "load_position": [0.2, 0.4, 0.6, np.nan],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})

    balance_engine_angles_arm = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array,
    )

    span_model_1 = deepcopy(
        balance_engine_angles_arm.balance_model.nodes_span_model
    )

    balance_engine_angles_arm.solve_adjustment()

    balance_engine_angles_arm.solve_change_state(new_temperature=75)

    assert len(
        balance_engine_angles_arm.balance_model.nodes_span_model.sagging_parameter
    ) > len(span_model_1.sagging_parameter)


def test_adjustment_convergence_error(monkeypatch, balance_engine_simple):
    def fail(*args):
        raise ConvergenceError("did not converge", origin="adjustment")

    # Patch at class level because solve_adjustment now recreates solver instances
    from mechaphlowers.core.models.balance.solvers.balance_solver import BalanceSolver
    monkeypatch.setattr(BalanceSolver, "solve", fail)

    with pytest.raises(ConvergenceError, match="did not converge"):
        balance_engine_simple.solve_adjustment()


def test_adjustment_convergence_error_origin(
    monkeypatch, balance_engine_simple
):
    # weird test: sets origin to "adjustment" but get replaced by "solve_adjustment" anyway in engine.py
    def fail_generator(origin: str):
        def fail(*args):
            raise ConvergenceError("did not converge", origin=origin)

        return fail

    # Patch at class level because solve_adjustment now recreates solver instances
    from mechaphlowers.core.models.balance.solvers.balance_solver import BalanceSolver
    monkeypatch.setattr(
        BalanceSolver,
        "solve",
        fail_generator("adjustment"),
    )

    with pytest.raises(ConvergenceError) as excinfo:
        balance_engine_simple.solve_adjustment()

    assert excinfo.value.origin == "solve_adjustment"
    assert getattr(excinfo.value, "origin", None) == "solve_adjustment"

    monkeypatch.setattr(
        balance_engine_simple.solver_change_state,
        "solve",
        fail_generator("change_state"),
    )
    # mocking L_ref and initial_L_ref to avoid launching adjustment solver first
    dummy_L_ref = np.zeros(balance_engine_simple.support_number - 1)
    balance_engine_simple.L_ref = dummy_L_ref
    balance_engine_simple.initial_L_ref = dummy_L_ref

    with pytest.raises(ConvergenceError) as excinfo:
        balance_engine_simple.solve_change_state()

    assert excinfo.value.origin == "solve_change_state"
    assert getattr(excinfo.value, "origin", None) == "solve_change_state"


def test_reset_restores_initial_state(balance_engine_simple: BalanceEngine):
    initial_span_param = (
        balance_engine_simple.span_model.sagging_parameter.copy()
    )
    initial_wind = balance_engine_simple.cable_loads.wind_pressure.copy()

    balance_engine_simple.span_model.sagging_parameter = np.ones_like(
        initial_span_param
    )
    balance_engine_simple.cable_loads.wind_pressure = np.ones_like(
        initial_wind
    )

    balance_engine_simple.reset(True)

    np.testing.assert_array_equal(
        balance_engine_simple.span_model.sagging_parameter, initial_span_param
    )
    np.testing.assert_array_equal(
        balance_engine_simple.cable_loads.wind_pressure, initial_wind
    )
    assert balance_engine_simple.balance_model.adjustment is True


def test_add_loads_wrong_values(balance_engine_simple: BalanceEngine):
    load_mass = np.array([500, 70, 0, np.nan])
    with pytest.raises(ValueError):
        balance_engine_simple.add_loads(
            np.array([-1, 200, 0, np.nan]), load_mass
        )
    with pytest.raises(ValueError):
        balance_engine_simple.add_loads(
            np.array([0, 1000, 0, np.nan]), load_mass
        )


def test_get_data_spans(balance_engine_simple: BalanceEngine):
    balance_engine_simple.solve_adjustment()
    balance_engine_simple.solve_change_state()
    data_spans = balance_engine_simple.get_data_spans()
    assert {
        'span_length',
        'elevation',
        'slope_left',
        'slope_right',
        'parameter',
        'tension_sup',
        'tension_inf',
        'L0',
        'horizontal_distance',
        'arc_length',
        'T_h',
        'sag',
        'sag_s2',
    } <= data_spans.keys()
    for value in data_spans.values():
        assert len(value) == 3


def test_get_data_spans_with_loads(balance_engine_simple: BalanceEngine):
    balance_engine_simple.add_loads(
        load_position_distance=[150, 200, 0, np.nan],
        load_mass=[200, 500, 0, np.nan],
    )
    balance_engine_simple.solve_adjustment()
    balance_engine_simple.solve_change_state()
    data_spans = balance_engine_simple.get_data_spans()
    for value in data_spans.values():
        assert len(value) == 3


def test_engine_wind_sense(balance_engine_simple: BalanceEngine):
    balance_engine_simple.solve_adjustment()

    # Test with wind_sense "clockwise"
    balance_engine_simple.solve_change_state(
        wind_pressure=200,
        wind_sense="clockwise",
    )
    displacement_clockwise = (
        balance_engine_simple.balance_model.chain_displacement()
    )

    np.testing.assert_array_equal(
        balance_engine_simple.balance_model.cable_loads.wind_pressure,
        np.array([-200.0, -200.0, -200.0, -200.0]),
    )

    # Test with wind_sense "anticlockwise"
    balance_engine_simple.solve_change_state(
        wind_pressure=-200,
        wind_sense="anticlockwise",
    )
    np.testing.assert_array_equal(
        balance_engine_simple.balance_model.cable_loads.wind_pressure,
        np.array([-200.0, -200.0, -200.0, -200.0]),
    )
    displacement_anticlockwise = (
        balance_engine_simple.balance_model.chain_displacement()
    )

    np.testing.assert_array_equal(
        displacement_clockwise, displacement_anticlockwise
    )


# ── performance tests ────────────────────────────────────────────────────────

def _make_8support_section_array(cable_array: "CableArray") -> "BalanceEngine":
    """8-support line with spans of varying length."""
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4", "5", "6", "7", "8"],
                "suspension": [False, True, True, True, True, True, True, False],
                "conductor_attachment_altitude": [
                    30.0, 45.0, 55.0, 60.0, 50.0, 65.0, 40.0, 35.0
                ],
                "crossarm_length": [0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0],
                "line_angle": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "insulator_length": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                "span_length": [400.0, 350.0, 450.0, 300.0, 500.0, 380.0, 420.0, np.nan],
                "insulator_mass": [
                    1000.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 1000.0
                ],
                "load_mass": [0.0] * 8,
                "load_position": [0.0] * 8,
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    return BalanceEngine(cable_array=cable_array, section_array=section_array)


def _make_12support_section_array(cable_array: "CableArray") -> "BalanceEngine":
    """12-support plain line for size-scaling comparison."""
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                "suspension": [
                    False, True, True, True, True, True,
                    True, True, True, True, True, False,
                ],
                "conductor_attachment_altitude": [
                    30.0, 45.0, 55.0, 60.0, 50.0, 65.0,
                    40.0, 35.0, 50.0, 58.0, 42.0, 38.0,
                ],
                "crossarm_length": [0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0],
                "line_angle": [0.0] * 12,
                "insulator_length": [3.0] * 12,
                "span_length": [
                    400.0, 350.0, 450.0, 300.0, 500.0, 380.0,
                    420.0, 370.0, 410.0, 340.0, 460.0, np.nan,
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
    return BalanceEngine(cable_array=cable_array, section_array=section_array)


@pytest.mark.benchmark
def test_perf_data_and_change_state_baseline_vs_manipulations(
    cable_array_AM600: "CableArray",
) -> None:
    """Compare .data and solve_change_state timing between:
    - a plain 8-support line (baseline),
    - the same 8-support line with 4 support manipulations, 1 rope manipulation
      and 4 virtual supports,
    - a plain 12-support line (size-scaling reference).

    Prints a timing table; does not assert on durations (benchmark only).
    """
    import time

    n_iterations = 20

    def _measure(engine: "BalanceEngine") -> tuple[float, float]:
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            _ = engine.section_array.data
        data_s = (time.perf_counter() - t0) / n_iterations

        t0 = time.perf_counter()
        for _ in range(n_iterations):
            engine.solve_change_state(new_temperature=15.0)
        change_state_s = (time.perf_counter() - t0) / n_iterations

        return data_s, change_state_s

    # ── baseline: plain 8-support ────────────────────────────────────────────
    engine_base = _make_8support_section_array(cable_array_AM600)
    engine_base.solve_adjustment()
    baseline_data_s, baseline_change_state_s = _measure(engine_base)

    # ── 8-support with manipulations ─────────────────────────────────────────
    engine_manip = _make_8support_section_array(cable_array_AM600)
    from mechaphlowers.core.manipulation import Manipulation

    manip = Manipulation(engine_manip.section_array)
    # 4 support manipulations (supports 1, 2, 4, 5)
    manip.support_manipulation({
        1: {"z": 1.0},
        2: {"z": -1.0, "y": 0.5},
        4: {"z": 2.0},
        5: {"y": -0.5},
    })
    # 1 rope manipulation (support 3)
    manip.rope_manipulation({3: 4.5})
    # 4 virtual supports (one per span: spans 0, 2, 4, 6)
    manip.add_virtual_support({
        0: {"x": 200.0, "y": 0.0, "z": 38.0, "insulator_length": 3.0, "insulator_mass": 500.0, "hanging_cable_point_from_left_support": 200.0},
        2: {"x": 200.0, "y": 0.0, "z": 58.0, "insulator_length": 3.0, "insulator_mass": 500.0, "hanging_cable_point_from_left_support": 200.0},
        4: {"x": 250.0, "y": 0.0, "z": 52.0, "insulator_length": 3.0, "insulator_mass": 500.0, "hanging_cable_point_from_left_support": 250.0},
        6: {"x": 200.0, "y": 0.0, "z": 42.0, "insulator_length": 3.0, "insulator_mass": 500.0, "hanging_cable_point_from_left_support": 200.0},
    })
    # Apply manipulation and rebuild engine
    manipulated_sa = manip.from_section_array(engine_manip.section_array)
    engine_manip = BalanceEngine(cable_array=cable_array_AM600, section_array=manipulated_sa)
    engine_manip.solve_adjustment()
    manip_data_s, manip_change_state_s = _measure(engine_manip)

    # ── size-scaling reference: plain 12-support ──────────────────────────────
    engine_12 = _make_12support_section_array(cable_array_AM600)
    engine_12.solve_adjustment()
    ref12_data_s, ref12_change_state_s = _measure(engine_12)

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
        ("solve_change_state", baseline_change_state_s, manip_change_state_s, ref12_change_state_s),
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


