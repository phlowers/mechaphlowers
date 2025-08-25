# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Callable, TypedDict

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.api.frames import SectionDataFrame
from mechaphlowers.core.solver.cable_state import (
    SagTensionSolver,
)
from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
    WeatherArray,
)
from mechaphlowers.entities.data_container import (
    factory_data_container,
)


class WeatherDict(TypedDict, total=False):
    ice_thickness: np.ndarray
    wind_pressure: np.ndarray


def create_sag_tension_solver(
    section_array: SectionArray,
    cable_array: CableArray,
    weather_array: WeatherArray,
) -> SagTensionSolver:
    frame = SectionDataFrame(section_array)
    frame.add_cable(cable_array)
    frame.add_weather(weather_array)

    data_container = factory_data_container(
        section_array, cable_array, weather_array
    )
    solver = SagTensionSolver(**data_container.__dict__)
    solver.initial_state()
    return solver


def test_functions_to_solve__same_loads(
    default_section_array_three_spans: SectionArray,
    default_cable_array: CableArray,
) -> None:
    NB_SPAN = 4
    frame = SectionDataFrame(default_section_array_three_spans)
    frame.add_cable(default_cable_array)

    weather_dict: WeatherDict = {
        "ice_thickness": np.zeros(NB_SPAN),
        "wind_pressure": np.zeros(NB_SPAN),
    }

    weather_array = WeatherArray(pd.DataFrame(weather_dict))

    frame.add_weather(weather_array)

    data_container = factory_data_container(
        default_section_array_three_spans, default_cable_array, weather_array
    )

    sag_tension_calculation = SagTensionSolver(**data_container.__dict__)
    sag_tension_calculation.initial_state()

    new_temperature = np.array([15] * NB_SPAN)
    sag_tension_calculation.change_state(
        **weather_dict,
        new_temperature=new_temperature,
    )
    T_h_state_0 = sag_tension_calculation.T_h_after_change

    # TODO: change this test after fixing the issue with NaN at last value
    assert (((T_h_state_0 - frame.span.T_h())[0:-1]) < 1e-6).all()  # type: ignore[union-attr]

    assert (
        sag_tension_calculation.p_after_change()[0]
        - default_section_array_three_spans.sagging_parameter
        < 1e-6
    )
    expected_p = np.array(
        [default_section_array_three_spans.sagging_parameter] * 3 + [np.nan]
    )
    np.testing.assert_allclose(
        sag_tension_calculation.p_after_change(), expected_p, atol=1e-5
    )

    np.testing.assert_allclose(
        sag_tension_calculation.L_after_change(),
        frame.span.L(),  # type: ignore[union-attr]
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "weather, temperature, expected_T_h, expected_L, expected_p",
    [
        (
            {
                "ice_thickness": np.array([0.0, 0.0]),
                "wind_pressure": np.array([0.0, 0.0]),
            },
            np.array([15, 15]),
            np.array([19109.88, np.nan]),
            np.array([481.25673597, np.nan]),
            np.array([2000, np.nan]),
        ),
        (
            {
                "ice_thickness": 2e-2 * np.ones(2),
                "wind_pressure": np.array([0.0, 0.0]),
            },
            np.array([15, 15]),
            np.array([42098.906999, np.nan]),
            np.array([481.80145209, np.nan]),
            np.array([1648.3929652, np.nan]),
        ),
        (
            {
                "ice_thickness": 1e-2 * np.ones(2),
                "wind_pressure": 200 * np.ones(2),
            },
            np.array([15, 15]),
            np.array([31745.051094, np.nan]),
            np.array([481.55593028, np.nan]),
            np.array([1782.3759651, np.nan]),
        ),
        (
            {
                "ice_thickness": np.array([0.0, 0.0]),
                "wind_pressure": np.array([0.0, 0.0]),
            },
            np.array([25, 25]),
            np.array([18380.111575, np.nan]),
            np.array([481.35015024, np.nan]),
            np.array([1923.6239657, np.nan]),
        ),
    ],
)
def test_functions_to_solve__different_weather(
    default_section_array_one_span: SectionArray,
    default_cable_array: CableArray,
    factory_neutral_weather_array: Callable[[int], WeatherArray],
    weather: dict,
    temperature: np.ndarray,
    expected_T_h: np.ndarray,
    expected_L: np.ndarray,
    expected_p: np.ndarray,
) -> None:
    initial_weather = factory_neutral_weather_array(2)

    sag_tension_calculation = create_sag_tension_solver(
        default_section_array_one_span,
        default_cable_array,
        initial_weather,
    )
    sag_tension_calculation.change_state(
        **weather, new_temperature=temperature
    )
    T_h = sag_tension_calculation.T_h_after_change
    assert T_h is not None
    np.testing.assert_allclose(T_h, expected_T_h, atol=1e-5)
    np.testing.assert_allclose(
        sag_tension_calculation.L_after_change(), expected_L, atol=1e-5
    )
    np.testing.assert_allclose(
        sag_tension_calculation.p_after_change(), expected_p, atol=1e-5
    )


def test_functions_to_solve__different_temp_ref(
    default_section_array_one_span: SectionArray,
    default_cable_array: CableArray,
    factory_neutral_weather_array: Callable[[int], WeatherArray],
) -> None:
    NB_SPAN = 2
    initial_weather_array = factory_neutral_weather_array(2)
    new_temperature = np.array([15] * NB_SPAN)

    sag_tension_calculation_0 = create_sag_tension_solver(
        default_section_array_one_span,
        default_cable_array,
        initial_weather_array,
    )

    weather_dict_final: WeatherDict = {
        "ice_thickness": 6e-2 * np.ones(NB_SPAN),
        "wind_pressure": 0 * np.ones(NB_SPAN),
    }

    sag_tension_calculation_0.change_state(
        **weather_dict_final, new_temperature=new_temperature
    )
    T_h_state_0 = sag_tension_calculation_0.T_h_after_change
    expected_result_0 = np.array([117951.847, np.nan])
    assert T_h_state_0 is not None
    np.testing.assert_allclose(T_h_state_0, expected_result_0, atol=0.01)

    new_cable_array = default_cable_array.__copy__()
    new_cable_array._data.temperature_reference = 0

    sag_tension_calculation_1 = create_sag_tension_solver(
        default_section_array_one_span,
        new_cable_array,
        initial_weather_array,
    )
    sag_tension_calculation_1.change_state(
        **weather_dict_final, new_temperature=new_temperature
    )
    T_h_state_1 = sag_tension_calculation_1.T_h_after_change
    expected_result_1 = np.array([117961.6142, np.nan])
    assert T_h_state_1 is not None
    np.testing.assert_allclose(T_h_state_1, expected_result_1, atol=0.01)


def test_functions_to_solve__no_memory_effect(
    default_section_array_one_span: SectionArray,
    default_cable_array: CableArray,
    factory_neutral_weather_array: Callable[[int], WeatherArray],
) -> None:
    """Create two solvers.
    One that goest through state 0 then state 1.
    An other that goes through state 1.
    Results should be the same for both solvers.
    """
    initial_weather = factory_neutral_weather_array(2)

    sag_tension_calculation_indirect = create_sag_tension_solver(
        default_section_array_one_span,
        default_cable_array,
        initial_weather,
    )

    weather_0: WeatherDict = {
        "ice_thickness": 1e-2 * np.ones(2),
        "wind_pressure": 200 * np.ones(2),
    }
    temperature_0 = np.array([25, 25])

    weather_1: WeatherDict = {
        "ice_thickness": 2e-2 * np.ones(2),
        "wind_pressure": 400 * np.ones(2),
    }
    temperature_1 = np.array([20, 20])

    sag_tension_calculation_indirect.change_state(
        **weather_0, new_temperature=temperature_0
    )
    sag_tension_calculation_indirect.change_state(
        **weather_1, new_temperature=temperature_1
    )
    T_h_indirect = sag_tension_calculation_indirect.T_h_after_change
    p_indirect = sag_tension_calculation_indirect.p_after_change()
    L_indirect = sag_tension_calculation_indirect.L_after_change()

    sag_tension_calculation_direct = create_sag_tension_solver(
        default_section_array_one_span,
        default_cable_array,
        initial_weather,
    )

    sag_tension_calculation_direct.change_state(
        **weather_1, new_temperature=temperature_1
    )
    T_h_direct = sag_tension_calculation_direct.T_h_after_change
    p_direct = sag_tension_calculation_direct.p_after_change()
    L_direct = sag_tension_calculation_direct.L_after_change()
    assert T_h_indirect is not None
    assert T_h_direct is not None
    np.testing.assert_allclose(T_h_indirect, T_h_direct)
    np.testing.assert_allclose(p_indirect, p_direct)
    np.testing.assert_allclose(L_indirect, L_direct)


def test_functions_to_solve__reset_to_initial_state(
    default_section_array_one_span: SectionArray,
    default_cable_array: CableArray,
    factory_neutral_weather_array: Callable[[int], WeatherArray],
) -> None:
    """Test that checks that going back to initial state after an intermedial state, gives the same result:
    initial state -> weather state -> initial state
    The two initial states should have the same result.
    """
    initial_weather = factory_neutral_weather_array(2)

    sag_tension_calculation = create_sag_tension_solver(
        default_section_array_one_span,
        default_cable_array,
        initial_weather,
    )
    T_h_initial_state_0 = sag_tension_calculation.T_h_after_change
    p_initial_state_0 = sag_tension_calculation.p_after_change()
    L_initial_state_0 = sag_tension_calculation.L_after_change()

    weather: WeatherDict = {
        "ice_thickness": 2e-2 * np.ones(2),
        "wind_pressure": 400 * np.ones(2),
    }
    temperature = np.array([20, 20])

    sag_tension_calculation.change_state(
        **weather, new_temperature=temperature
    )

    sag_tension_calculation.initial_state()
    T_h_initial_state_1 = sag_tension_calculation.T_h_after_change
    p_initial_state_1 = sag_tension_calculation.p_after_change()
    L_initial_state_1 = sag_tension_calculation.L_after_change()
    assert T_h_initial_state_0 is not None
    assert T_h_initial_state_1 is not None
    np.testing.assert_allclose(T_h_initial_state_0, T_h_initial_state_1)
    np.testing.assert_allclose(p_initial_state_0, p_initial_state_1)
    np.testing.assert_allclose(L_initial_state_0, L_initial_state_1)
