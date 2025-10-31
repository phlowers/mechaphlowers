# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd
import pytest
from pytest import fixture

from mechaphlowers.core.models.balance.engine import (
    BalanceEngine,
)
from mechaphlowers.entities.arrays import CableArray, SectionArray


@fixture
def cable_array_AM600() -> CableArray:
    return CableArray(
        pd.DataFrame(
            {
                "section": [600.4],
                "diameter": [31.86],
                "linear_mass": [1.8],
                "young_modulus": [60],
                "dilatation_coefficient": [23],
                "temperature_reference": [15],
                "a0": [0],
                "a1": [60],
                "a2": [0],
                "a3": [0],
                "a4": [0],
                "b0": [0],
                "b1": [0],
                "b2": [0],
                "b3": [0],
                "b4": [0],
            }
        )
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
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
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


def test_element_change_state(balance_engine_simple: BalanceEngine):
    with pytest.raises(AttributeError):
        balance_engine_simple.solve_change_state()

    balance_engine_simple.solve_adjustment()

    balance_engine_simple.solve_change_state()


def test_change_state_size(balance_engine_simple: BalanceEngine):
    balance_engine_simple.solve_adjustment()
    with pytest.raises(AttributeError):
        balance_engine_simple.solve_change_state(
            wind_pressure=np.array([1, 1, 1])
        )  # array size should be 4
    with pytest.raises(AttributeError):
        balance_engine_simple.solve_change_state(
            ice_thickness=np.array([1, 1, 1])
        )
    with pytest.raises(AttributeError):
        balance_engine_simple.solve_change_state(
            new_temperature=np.array([1, 1, 1])
        )


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

    # Test with only wind_pressure provided
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
