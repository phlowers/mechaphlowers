# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mechaphlowers.config import options as options
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
)
from mechaphlowers.plotting.plot import (
    PlotEngine,
)
from test.tools.plot_tools import assert_cable_linked_to_attachment


@pytest.fixture
def balance_engine_no_loads(cable_array_AM600: CableArray) -> BalanceEngine:
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
                "load_mass": [0, 0, 0, np.nan],
                "load_position": [0, 0, 0, np.nan],
            }
        )
    )
    section_array.add_units({"line_angle": "grad"})

    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    return BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array,
    )


@pytest.fixture
def balance_engine_with_loads(cable_array_AM600: CableArray) -> BalanceEngine:
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
                "load_mass": [500, 0, 1000, np.nan],
                "load_position": [0.2, 0.4, 0.6, np.nan],
            }
        )
    )
    section_array.add_units({"line_angle": "grad"})

    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    return BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array,
    )


def test_plot_loads(balance_engine_with_loads: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_with_loads
    )

    balance_engine_with_loads.solve_adjustment()
    balance_engine_with_loads.solve_change_state(
        new_temperature=15, wind_pressure=560
    )

    fig = go.Figure()

    plt_engine.preview_line3d(fig)

    # fig.show()

    span_points, _, insulators_points = plt_engine.get_points_for_plot()
    assert_cable_linked_to_attachment(span_points, insulators_points)


def test_plot_add_loads_later(balance_engine_no_loads: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_no_loads
    )

    balance_engine_no_loads.solve_adjustment()

    # Modify loads positions and mass
    balance_engine_no_loads.section_array._data["load_mass"] = [
        5000,
        10000,
        0,
        np.nan,
    ]
    balance_engine_no_loads.section_array._data["load_position"] = [
        0.2,
        0.4,
        0.7,
        np.nan,
    ]

    # Reset objects to factor in modifications
    balance_engine_no_loads.reset()
    plt_engine = plt_engine.generate_reset()

    balance_engine_no_loads.solve_adjustment()
    balance_engine_no_loads.solve_change_state(
        new_temperature=15, wind_pressure=560
    )
    assert plt_engine.spans.span_length.size == 6
    fig = go.Figure()

    plt_engine.preview_line3d(fig)

    # fig.show()

    span_points, _, insulators_points = (
        plt_engine.section_pts.get_points_for_plot()
    )
    assert_cable_linked_to_attachment(span_points, insulators_points)


def test_plot_reset(balance_engine_no_loads: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_no_loads
    )

    balance_engine_no_loads.reset()
    plt_engine = plt_engine.generate_reset()

    # Checks that id are still the same
    assert id(balance_engine_no_loads.balance_model.nodes_span_model) == id(
        plt_engine.spans
    )
    assert id(balance_engine_no_loads.cable_loads) == id(
        plt_engine.cable_loads
    )


def test_plot_add_loads(balance_engine_no_loads: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_no_loads
    )

    # Modify loads positions and mass
    balance_engine_no_loads.add_loads(
        load_position_distance=np.array([100, 150, 300, np.nan]),
        load_mass=[5000, 10000, 0, np.nan],
    )

    # Reset objects to factor in modifications
    plt_engine = plt_engine.generate_reset()

    balance_engine_no_loads.solve_adjustment()
    balance_engine_no_loads.solve_change_state(
        new_temperature=15, wind_pressure=560
    )
    assert plt_engine.spans.span_length.size == 6

    fig = go.Figure()

    plt_engine.preview_line3d(fig)

    # fig.show()

    span_points, _, insulators_points = (
        plt_engine.section_pts.get_points_for_plot()
    )
    assert_cable_linked_to_attachment(span_points, insulators_points)


def test_get_loads_coords(balance_engine_with_loads: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_with_loads
    )
    coords_loads_before_solve = plt_engine.get_loads_coords()
    assert coords_loads_before_solve == {}

    balance_engine_with_loads.solve_adjustment()
    balance_engine_with_loads.solve_change_state(
        new_temperature=15, wind_pressure=560
    )

    coords_loads = plt_engine.get_loads_coords()
    assert len(coords_loads) == 2
    assert list(coords_loads.keys()) == [0, 2]
    # fig = go.Figure()
    # plt_engine.preview_line3d(fig)
    # fig.show()


def test_get_loads_coords_4_spans(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4", "5"],
                "suspension": [False, True, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65, 40],
                "crossarm_length": [0, 10, -10, 0, 0],
                "line_angle": [0, 10, 0, 0, 0],
                "insulator_length": [3, 3, 3, 3, 3],
                "span_length": [500, 300, 400, 400, np.nan],
                "insulator_mass": [100, 50, 50, 50, 100],
                "load_mass": [500, 1000, 0, 0, np.nan],
                "load_position": [0.2, 0.4, 0.6, 0, np.nan],
            }
        )
    )
    section_array.add_units({"line_angle": "grad"})

    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array,
    )
    plt_engine = PlotEngine.builder_from_balance_engine(balance_engine)
    coords_loads_before_solve = plt_engine.get_loads_coords()
    assert coords_loads_before_solve == {}

    balance_engine.solve_adjustment()
    balance_engine.solve_change_state(new_temperature=15)

    coords_loads = plt_engine.get_loads_coords()
    assert len(coords_loads) == 2
    assert list(coords_loads.keys()) == [0, 1]
    # fig = go.Figure()
    # plt_engine.preview_line3d(fig)
    # fig.show()


def test_get_coords_no_loads(balance_engine_no_loads: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_no_loads
    )

    balance_engine_no_loads.solve_adjustment()

    # Modify loads positions and mass
    balance_engine_no_loads.section_array._data["load_mass"] = [
        5000,
        10000,
        0,
        np.nan,
    ]
    balance_engine_no_loads.section_array._data["load_position"] = [
        0.2,
        0.4,
        0.7,
        np.nan,
    ]

    # Reset objects to factor in modifications
    balance_engine_no_loads.reset()
    plt_engine = plt_engine.generate_reset()

    balance_engine_no_loads.solve_adjustment()
    balance_engine_no_loads.solve_change_state(
        new_temperature=15, wind_pressure=560
    )
    assert plt_engine.spans.span_length.size == 6
    fig = go.Figure()

    plt_engine.preview_line3d(fig)

    # fig.show()

    span_points, _, insulators_points = (
        plt_engine.section_pts.get_points_for_plot()
    )
    assert_cable_linked_to_attachment(span_points, insulators_points)
