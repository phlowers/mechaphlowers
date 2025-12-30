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
from mechaphlowers.entities.errors import ConvergenceError


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
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
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
        raise ConvergenceError("did not converge", level="adjustment")

    # Mock the solver to raise
    monkeypatch.setattr(balance_engine_simple.solver_adjustment, "solve", fail)

    with pytest.raises(ConvergenceError, match="did not converge"):
        balance_engine_simple.solve_adjustment()


def test_adjustment_convergence_error_origin(monkeypatch, balance_engine_simple):
    
    def fail_generator(level: str):
        def fail(_: object):
            raise ConvergenceError("did not converge", level=level)
        return fail

    monkeypatch.setattr(balance_engine_simple.solver_adjustment, "solve", fail_generator("adjustment"))

    with pytest.raises(ConvergenceError) as excinfo:
        balance_engine_simple.solve_adjustment()

    assert excinfo.value.level == "adjustment"
    assert getattr(excinfo.value, "origin", None) == "solve_adjustment"
    
    monkeypatch.setattr(balance_engine_simple.solver_change_state, "solve", fail_generator("change_state"))
    # mocking L_ref to avoid launching adjustment solver first
    balance_engine_simple.L_ref = 500.0

    with pytest.raises(ConvergenceError) as excinfo:
        balance_engine_simple.solve_change_state()

    assert excinfo.value.level == "change_state"
    assert getattr(excinfo.value, "origin", None) == "solve_change_state"


def test_reset_restores_initial_state(balance_engine_simple: BalanceEngine):
    initial_span_param = balance_engine_simple.span_model.sagging_parameter.copy()
    initial_wind = balance_engine_simple.cable_loads.wind_pressure.copy()

    balance_engine_simple.span_model.sagging_parameter = np.ones_like(
        initial_span_param
    )
    balance_engine_simple.cable_loads.wind_pressure = np.ones_like(
        initial_wind
    )

    balance_engine_simple.reset()

    np.testing.assert_array_equal(
        balance_engine_simple.span_model.sagging_parameter, initial_span_param
    )
    np.testing.assert_array_equal(
        balance_engine_simple.cable_loads.wind_pressure, initial_wind
    )
    assert balance_engine_simple.balance_model.adjustment is True
    
    


