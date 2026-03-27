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
    BalanceEngineWarning,
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
        )
    )
    section_array.add_units({"line_angle": "grad"})
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
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
        )
    )
    section_array.add_units({"line_angle": "grad"})

    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

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
        )
    )
    section_array.add_units({"line_angle": "grad"})

    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

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
    def fail(_: object):
        raise ConvergenceError("did not converge", origin="adjustment")

    # Mock the solver to raise
    monkeypatch.setattr(balance_engine_simple.solver_adjustment, "solve", fail)

    with pytest.raises(ConvergenceError, match="did not converge"):
        balance_engine_simple.solve_adjustment()


def test_adjustment_convergence_error_origin(
    monkeypatch, balance_engine_simple
):
    # weird test: sets origin to "adjustment" but get replaced by "solve_adjustment" anyway in engine.py
    def fail_generator(origin: str):
        def fail(_: object):
            raise ConvergenceError("did not converge", origin=origin)

        return fail

    monkeypatch.setattr(
        balance_engine_simple.solver_adjustment,
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
        'parameter',
        'tension_sup',
        'tension_inf',
        'L0',
        'horizontal_distance',
        'arc_length',
        'T_h',
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


@pytest.mark.integration
def test_shifting_and_shortening_cable(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [5, 10, -10, 5],
                "line_angle": [0, 30, 0, 0],
                "insulator_length": [0.01, 3, 3, 0.01],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [1000, 500, 500, 1000],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        ),
        sagging_parameter=1200,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600, section_array=section_array
    )

    # Base case: no shifting, no shortening

    with pytest.warns(BalanceEngineWarning):
        balance_engine.solve_change_state(
            wind_pressure=0.0, new_temperature=15.0
        )

    assert balance_engine.L_ref.shape == (3,)
    np.testing.assert_allclose(
        balance_engine.span_model.T_h(),
        np.array([2119.0, 2119.0, 2119.0, np.nan]) * 10,
        atol=10,
    )
    np.testing.assert_allclose(
        balance_engine.L_ref, np.array([500.8, 298.5, 401.7]), atol=0.1
    )

    # Shift support 2 by 1m
    balance_engine.add_cable_shifting(shift_support=np.array([0, 1, 0, 0]))
    balance_engine.shift_shorten_cable()

    assert balance_engine.L_ref.shape == (3,)
    np.testing.assert_allclose(
        balance_engine.L_ref, np.array([501.8, 297.5, 401.7]), atol=0.1
    )

    balance_engine.solve_change_state(wind_pressure=0.0, new_temperature=15.0)
    np.testing.assert_allclose(
        balance_engine.span_model.T_h(),
        np.array([2026.0, 2315.0, 2246.0, np.nan]) * 10.0,
        atol=10,
    )

    # shorten span 2 by 2m
    balance_engine.add_cable_shifting(shorten_span=np.array([0, 2, 0]))
    balance_engine.shift_shorten_cable()

    np.testing.assert_allclose(
        balance_engine.L_ref, np.array([500.8, 296.5, 401.7]), atol=0.1
    )

    balance_engine.solve_change_state(wind_pressure=0.0, new_temperature=15.0)
    np.testing.assert_allclose(
        balance_engine.span_model.T_h(),
        np.array([2411.0, 2846.0, 2614.0, np.nan]) * 10.0,
        atol=10,
    )

    # Shift support 2 by 1.5m and shorten span 1 by 1.5m
    balance_engine.add_cable_shifting(
        shorten_span=np.array([1.5, 0, 0]),
        shift_support=np.array([0, 1, 0.5, 0]),
    )
    balance_engine.shift_shorten_cable()

    np.testing.assert_allclose(
        balance_engine.L_ref, np.array([500.338, 298.042, 401.254]), atol=0.1
    )

    balance_engine.solve_change_state(wind_pressure=0.0, new_temperature=15.0)
    np.testing.assert_allclose(
        balance_engine.span_model.T_h(),
        np.array([2353.0, 2459.0, 2454.0, np.nan]) * 10.0,
        atol=10,
    )
    np.testing.assert_allclose(
        balance_engine.parameter,
        np.array([1333.0, 1392.0, 1390.0, np.nan]),
        atol=1,
    )


def test_add_cable_shifting_default_values(
    balance_engine_simple: BalanceEngine,
):
    balance_engine_simple.add_cable_shifting()

    expected_support = np.zeros(balance_engine_simple.support_number)
    np.testing.assert_array_equal(
        balance_engine_simple.shift_support, expected_support
    )
    expected_span = np.zeros(balance_engine_simple.support_number - 1)
    np.testing.assert_array_equal(
        balance_engine_simple.shortening_span, expected_span
    )


def test_add_cable_shifting_wrong_size_shifting(
    balance_engine_simple: BalanceEngine,
):
    with pytest.raises(ValueError, match="shift_support has incorrect size"):
        balance_engine_simple.add_cable_shifting(
            shift_support=np.array([0.0, 1.0, 0.0])  # 3 elements, 4 expected
        )


def test_add_cable_shifting_wrong_size_shortening(
    balance_engine_simple: BalanceEngine,
):
    with pytest.raises(ValueError, match="shorten_span has incorrect size"):
        balance_engine_simple.add_cable_shifting(
            shorten_span=np.array(
                [0.0, 1.0, 0.0, 0.0]
            )  # 4 elements, 3 expected
        )


def test_add_cable_shifting_enforces_shifting_boundaries(
    balance_engine_simple: BalanceEngine,
):
    with pytest.warns(
        BalanceEngineWarning,
        match="shift_support first and last values have been set to 0",
    ):
        balance_engine_simple.add_cable_shifting(
            shift_support=np.array([5.0, 1.0, 2.0, 3.0])
        )

    assert balance_engine_simple.shift_support[0] == 0.0
    assert balance_engine_simple.shift_support[-1] == 0.0
    np.testing.assert_array_equal(
        balance_engine_simple.shift_support[1:-1],
        np.array([1.0, 2.0]),
    )


def test_add_cable_shifting_no_warning_when_boundaries_are_compliant(
    balance_engine_simple: BalanceEngine,
):
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", BalanceEngineWarning)
        balance_engine_simple.add_cable_shifting(
            shift_support=np.array([0.0, 1.0, 2.0, 0.0]),
            shorten_span=np.array([0.0, 1.0, 2.0]),
        )


def test_add_cable_shifting_stores_values(
    balance_engine_simple: BalanceEngine,
):
    shifting = np.array([0.0, 1.5, 2.0, 0.0])
    shortening = np.array([0.0, 0.5, 1.0])

    balance_engine_simple.add_cable_shifting(
        shift_support=shifting,
        shorten_span=shortening,
    )

    np.testing.assert_array_equal(
        balance_engine_simple.shift_support, shifting
    )
    np.testing.assert_array_equal(
        balance_engine_simple.shortening_span, shortening
    )
