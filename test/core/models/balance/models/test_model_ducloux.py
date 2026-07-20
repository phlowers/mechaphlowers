# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd
import pytest

from mechaphlowers import sample_cable_catalog
from mechaphlowers.api.section_study import SectionStudy
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.core.models.balance.models.model_ducloux import (
    nodes_builder,
)
from mechaphlowers.core.models.balance.span_loads import SpanLoads
from mechaphlowers.entities.arrays import CableArray, SectionArray


def test_nodes_builder(
    section_array_complete: SectionArray,
    span_loads_for_complete_section_array: SpanLoads,
):
    nodes_builder(
        section_array_complete, span_loads_for_complete_section_array
    )


def test_load_span_model(cable_array_AM600: CableArray):
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
                "insulator_mass": [100, 50, 500, 0],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})

    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array,
    )
    balance_engine.set_loads(
        load_position_distance=[100, 120, 240],
        load_mass=[0, 500, 0],
    )
    balance_engine.solve_adjustment()

    balance_engine.solve_change_state()
    nodes_span_model = balance_engine.balance_model.nodes_span_model
    assert nodes_span_model.parameter.shape == (5,)
    np.testing.assert_equal(
        nodes_span_model.span_index, np.array([0, 1, 1, 2, 3])
    )
    np.testing.assert_equal(
        nodes_span_model.span_type, np.array([0, 1, 2, 0, 0])
    )


def test_solve_twice(cable_array_AM600, default_section_array_three_spans):
    # Test that calling solve_change_state twice
    # returns the same results twice
    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=default_section_array_three_spans,
    )

    balance_engine.solve_adjustment()

    balance_engine.solve_change_state()
    first_dxdydz = balance_engine.balance_model.nodes.dxdydz.copy()
    balance_engine.solve_change_state()
    second_dxdydz = balance_engine.balance_model.nodes.dxdydz.copy()

    np.testing.assert_allclose(
        second_dxdydz, first_dxdydz, rtol=1e-6, atol=1e-8
    )


def test_solve_twice_after_set_loads(
    cable_array_AM600, default_section_array_three_spans
):
    # Test that calling solve_change_state twice
    # returns the same results twice
    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=default_section_array_three_spans,
    )

    balance_engine.set_loads(np.array([0, 0, 200]), np.array([0, 0, 50]))

    balance_engine.solve_adjustment()

    balance_engine.solve_change_state()
    first_dxdydz = balance_engine.balance_model.nodes.dxdydz.copy()
    balance_engine.solve_change_state()
    second_dxdydz = balance_engine.balance_model.nodes.dxdydz.copy()

    np.testing.assert_allclose(
        second_dxdydz, first_dxdydz, rtol=1e-6, atol=1e-8
    )


def test_solve_twice_after_add_loads(
    cable_array_AM600, default_section_array_three_spans
):
    # Test that calling solve_change_state twice
    # returns the same results twice
    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=default_section_array_three_spans,
    )

    balance_engine.add_loads(
        np.array([0, 0, 200, np.nan]), np.array([0, 0, 50, np.nan])
    )

    balance_engine.solve_adjustment()

    balance_engine.solve_change_state()
    first_dxdydz = balance_engine.balance_model.nodes.dxdydz.copy()
    balance_engine.solve_change_state()
    second_dxdydz = balance_engine.balance_model.nodes.dxdydz.copy()

    np.testing.assert_allclose(
        second_dxdydz, first_dxdydz, rtol=1e-6, atol=1e-8
    )


def test_change_temperature_back_to_initial(
    cable_array_AM600, default_section_array_three_spans
):
    # Test that changing temperature (15°C -> 30°C -> 15°C)
    # brings the displacement back to its initial value
    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=default_section_array_three_spans,
    )

    balance_engine.solve_adjustment()

    balance_engine.solve_change_state(new_temperature=15)
    first_dxdydz_15 = balance_engine.balance_model.nodes.dxdydz.copy()

    balance_engine.solve_change_state(new_temperature=30)

    balance_engine.solve_change_state(new_temperature=15)
    second_dxdydz_15 = balance_engine.balance_model.nodes.dxdydz.copy()

    np.testing.assert_allclose(
        second_dxdydz_15,
        first_dxdydz_15,
        rtol=1e-6,
        atol=1e-8,
    )


class TestMultipleAdjustmentsSameL0:
    """Series of test to ensure that successive calls of solve_adjustment() and solve_change_state() gives the same L0"""

    @pytest.fixture
    def study_4span_no_load(self) -> SectionStudy:
        cable_array = sample_cable_catalog.get_as_object(["ASTER600"])
        section_array = SectionArray(
            pd.DataFrame(
                {
                    "name": ["1", "2", "3", "4", "5"],
                    "suspension": [False, True, True, True, False],
                    "conductor_attachment_altitude": [
                        50.0,
                        50.0,
                        50.0,
                        50.0,
                        50.0,
                    ],
                    "crossarm_length": [0.0, 5.0, 5.0, 5.0, 0.0],
                    "line_angle": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "insulator_length": [3.0, 3.0, 3.0, 3.0, 3.0],
                    "span_length": [400.0, 400.0, 400.0, 400.0, np.nan],
                    "insulator_mass": [1000.0, 500.0, 500.0, 500.0, 1000.0],
                    "load_mass": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "load_position": [0.0, 0.0, 0.0, 0.0, 0.0],
                }
            ),
            sagging_parameter=2000,
            sagging_temperature=15,
        )
        section_array.add_units({"line_angle": "grad"})
        study = SectionStudy(
            cable_array=cable_array, section_array=section_array
        )
        return study

    def test_ice_change_state_same_L0(
        self, study_4span_no_load: SectionStudy
    ) -> None:
        study_4span_no_load.solve_adjustment()
        study_4span_no_load.solve_change_state()
        data = study_4span_no_load.get_data_spans()
        initial_L0 = data["L0"]

        study_4span_no_load.solve_adjustment()
        data = study_4span_no_load.get_data_spans()
        L0_1 = data["L0"]
        study_4span_no_load.solve_change_state(ice_thickness=0.1)
        np.testing.assert_allclose(initial_L0, L0_1)

        study_4span_no_load.solve_adjustment()
        data = study_4span_no_load.get_data_spans()
        L0_2 = data["L0"]
        np.testing.assert_allclose(initial_L0, L0_2)
        study_4span_no_load.solve_change_state(ice_thickness=0.1)

    def test_wind_change_state_same_L0(
        self, study_4span_no_load: SectionStudy
    ) -> None:
        study_4span_no_load.solve_adjustment()
        study_4span_no_load.solve_change_state()
        data = study_4span_no_load.get_data_spans()
        initial_L0 = data["L0"]

        study_4span_no_load.solve_adjustment()
        data = study_4span_no_load.get_data_spans()
        L0_1 = data["L0"]
        study_4span_no_load.solve_change_state(wind_pressure=500)
        np.testing.assert_allclose(initial_L0, L0_1)

        study_4span_no_load.solve_adjustment()
        data = study_4span_no_load.get_data_spans()
        L0_2 = data["L0"]
        np.testing.assert_allclose(initial_L0, L0_2)
        study_4span_no_load.solve_change_state(wind_pressure=500)

    def test_temperature_change_state_same_L0(
        self, study_4span_no_load: SectionStudy
    ) -> None:
        study_4span_no_load.solve_adjustment()
        study_4span_no_load.solve_change_state()
        data = study_4span_no_load.get_data_spans()
        initial_L0 = data["L0"]

        study_4span_no_load.solve_adjustment()
        data = study_4span_no_load.get_data_spans()
        L0_1 = data["L0"]
        study_4span_no_load.solve_change_state(new_temperature=60)
        np.testing.assert_allclose(initial_L0, L0_1)

        study_4span_no_load.solve_adjustment()
        data = study_4span_no_load.get_data_spans()
        L0_2 = data["L0"]
        np.testing.assert_allclose(initial_L0, L0_2)
        study_4span_no_load.solve_change_state(new_temperature=60)

    def test_add_successive_loads(self, study_4span_no_load: SectionStudy):
        study_4span_no_load.solve_adjustment()
        data = study_4span_no_load.get_data_spans()
        initial_L0 = data["L0"]

        study_4span_no_load.set_loads(
            load_position_distance=[200.0, 0.0, 0.0, 0.0],
            load_mass=[500.0, 0.0, 0.0, 0.0],
        )
        study_4span_no_load.solve_adjustment()
        data = study_4span_no_load.get_data_spans()
        L0_first_load = data["L0"]
        np.testing.assert_allclose(initial_L0, L0_first_load)

        # Second cycle: 2 loaded spans — stale cache expects 1 slot, crashes in build_merged
        study_4span_no_load.set_loads(
            load_position_distance=[200.0, 200.0, 0.0, 0.0],
            load_mass=[500.0, 300.0, 0.0, 0.0],
        )

        study_4span_no_load.solve_adjustment()
        data = study_4span_no_load.get_data_spans()
        L0_second_load = data["L0"]
        np.testing.assert_allclose(initial_L0, L0_second_load)


@pytest.fixture
def study_4span_with_load() -> SectionStudy:
    cable_array = sample_cable_catalog.get_as_object(["ASTER600"])
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4", "5"],
                "suspension": [False, True, True, True, False],
                "conductor_attachment_altitude": [
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                    50.0,
                ],
                "crossarm_length": [0.0, 5.0, 5.0, 5.0, 0.0],
                "line_angle": [0.0, 0.0, 0.0, 0.0, 0.0],
                "insulator_length": [3.0, 3.0, 3.0, 3.0, 3.0],
                "span_length": [400.0, 400.0, 400.0, 400.0, np.nan],
                "insulator_mass": [1000.0, 500.0, 500.0, 500.0, 1000.0],
                "load_mass": [0.0, 0.0, 0.0, 0.0, 0.0],
                "load_position": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    study = SectionStudy(cable_array=cable_array, section_array=section_array)
    study.set_loads(
        load_position_distance=[200.0, 0.0, 0.0, 0.0],
        load_mass=[500.0, 0.0, 0.0, 0.0],
    )
    return study


def test_repeated_solve_idempotent_loads(
    study_4span_with_load: SectionStudy,
) -> None:
    study_4span_with_load.solve_adjustment()
    study_4span_with_load.solve_change_state(new_temperature=15)
    data_first = study_4span_with_load.get_data_spans()
    L0_first = data_first["L0"]
    for i in range(4):
        study_4span_with_load.solve_adjustment()
        study_4span_with_load.solve_change_state(new_temperature=60)
        data = study_4span_with_load.get_data_spans()
        current_L0 = data["L0"]
        np.testing.assert_allclose(
            current_L0, L0_first, err_msg=f"L0 drifted at iteration {i + 2}"
        )
