# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import copy

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytest

from mechaphlowers.api.frames import SectionDataFrame
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.data.units import Q_
from mechaphlowers.entities.arrays import (
    SectionArray,
    WeatherArray,
)
from mechaphlowers.entities.shapes import SupportShape
from mechaphlowers.plotting.plot import PlotLine, plot_support_shape

data = {
    "name": ["1", "2", "three", "support 4"],
    "suspension": [False, True, True, False],
    "conductor_attachment_altitude": [50.0, 40.0, 20.0, 10.0],
    "crossarm_length": [
        5.0,
    ]
    * 4,
    "line_angle": [
        0,
    ]
    * 4,
    "insulator_length": [0, 4, 3.2, 0],
    "span_length": [100, 200, 300, np.nan],
    "insulator_weight": [1000.0, 500.0, 500.0, 1000.0],
}

section = SectionArray(data=pd.DataFrame(data))
section.sagging_parameter = 500
section.sagging_temperature = 15

frame = SectionDataFrame(section)


def test_plot_line3d__all_line() -> None:
    fig = go.Figure()
    frame.plot.line3d(fig)
    # fig.show() # deactivate for auto unit testing
    assert True  # Just trying to see if the previous code raises


def test_plot_line3d__subset() -> None:
    fig = go.Figure()
    frame.select(["1", "2"]).plot.line3d(fig)
    # fig.show() # deactivate for auto unit testing
    assert True  # Just trying to see if the previous code raises


def test_plot_line3d__view_option() -> None:
    fig = go.Figure()
    frame.plot.line3d(fig, view="full")
    assert True  # Just trying to see if the previous code raises


def test_plot_line3d__wrong_view_option() -> None:
    fig = go.Figure()
    with pytest.raises(ValueError):
        frame.plot.line3d(fig, view="wrong_parameter")
    with pytest.raises(ValueError):
        frame.plot.line3d(fig, view=22)


def test_plot_line3d__with_beta(
    default_cable_array,
):
    weather = WeatherArray(
        pd.DataFrame(
            {
                "ice_thickness": [0.0, 60.0, 0.0, np.nan],
                "wind_pressure": [240.12, 0.0, 600.0, np.nan],
            }
        )
    )
    frame.add_cable(cable=default_cable_array)
    frame.add_weather(weather=weather)
    frame.cable_loads.load_angle  # type: ignore[union-attr]
    fig = go.Figure()
    frame.plot.line3d(fig)
    # fig.show() # deactivate for auto unit testing
    assert True  # Just trying to see if the previous code raises


def test_plot_support_shape():
    fig = go.Figure()
    pyl_shape = SupportShape(
        name="pyl",
        yz_arms=np.array(
            [
                [0, 18.5],
                [3, 14.5],
                [6, 14.5],
                [9, 14.5],
                [-3, 14.5],
                [-6, 14.5],
                [-9, 14.5],
            ]
        ),
        set_number=np.array([22, 28, 37, 45, 46, 47, 55]),
    )
    plot_support_shape(fig, pyl_shape)
    # fig.show()
    assert True


def test_plot_flat_line3d(default_cable_array):
    # TODO: I dont know why those data gets balance Engine to crash. To investigate later.
    # data = {
    #     "name": ["1", "2", "three", "support 4"],
    #     "suspension": [False, True, True, False],
    #     "conductor_attachment_altitude": [50.0, 40.0, 20.0, 10.0],
    #     "crossarm_length": [
    #         5.0,
    #     ]
    #     * 4,
    #     "line_angle": [
    #         0,10,15,20
    #     ],
    #     "insulator_length": [0, 4, 3.2, 0],
    #     "span_length": [100, 200, 300, np.nan],
    #     "insulator_weight": [1000.0, 500.0, 500.0, 1000.0],
    #     "load_weight": [0, 0, 0, 0],
    #     "load_position": [0, 0, 0, 0],
    # }

    data = {
        "name": ["1", "2", "3", "4"],
        "suspension": [False, True, True, False],
        "conductor_attachment_altitude": [30, 30, 30, 30],
        "crossarm_length": [0, 10, -10, 0],
        "line_angle": Q_(np.array([0, 0, 0, 0]), "grad").to('deg').magnitude,
        "insulator_length": [3, 3, 3, 3],
        "span_length": [500, 300, 400, np.nan],
        "insulator_weight": [1000, 500, 500, 1000],
        "load_weight": [0, 0, 0, 0],
        "load_position": [0, 0, 0, 0],
    }

    section = SectionArray(data=pd.DataFrame(data))
    section.sagging_parameter = 500
    section.sagging_temperature = 15

    weather = WeatherArray(
        pd.DataFrame(
            {
                "ice_thickness": [0.0, 60.0, 0.0, np.nan],
                "wind_pressure": [240.12, 0.0, 600.0, np.nan],
            }
        )
    )

    be = BalanceEngine(section_array=section, cable_array=default_cable_array)

    plt_line = PlotLine.builder_from_balance_engine(be)

    fig = go.Figure()

    print(plt_line.beta)

    be.solve_adjustment()
    be.solve_change_state(wind_pressure=200 * np.array([1, 1, 1, np.nan]))

    print(plt_line.beta)  # check if beta is updated after change_state

    plt_line.preview_line3d(fig)

    fig.show()  # deactivate for auto unit testing
    assert True


def test_plot(balance_engine_base_test: BalanceEngine):
    plt_line = PlotLine.builder_from_balance_engine(balance_engine_base_test)
    balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1])
    )
    plt_line = PlotLine.builder_from_balance_engine(balance_engine_base_test)
    fig = go.Figure()
    plt_line.preview_line3d(fig)
    # balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(
        wind_pressure=1000 * np.array([1, 1, 1, np.nan]),
        # new_temperature=90 * np.array([1, 1, 1])
    )

    plt_line = PlotLine.builder_from_balance_engine(balance_engine_base_test)

    plt_line.preview_line3d(fig)

    plt_line_1 = copy.deepcopy(plt_line)

    # fig = go.Figure()

    print(plt_line.beta)
    # plt_line.preview_line3d(fig)

    # balance_engine_base_test.solve_adjustment()
    # balance_engine_base_test.solve_change_state(
    #     wind_pressure=200 * np.array([1, 1, 1, np.nan])
    # )

    # print(plt_line.beta)  # check if beta is updated after change_state

    # plt_line.preview_line3d(fig)

    fig.show()  # deactivate for auto unit testing
    assert True


def test_plot_ice(balance_engine_base_test: BalanceEngine):
    plt_line = PlotLine.builder_from_balance_engine(balance_engine_base_test)
    balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1])
    )
    plt_line = PlotLine.builder_from_balance_engine(balance_engine_base_test)
    fig = go.Figure()
    plt_line.preview_line3d(fig)
    # balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(
        ice_thickness=np.array([1, 1, 1, np.nan]),
        # new_temperature=90 * np.array([1, 1, 1]),
    )

    plt_line = PlotLine.builder_from_balance_engine(balance_engine_base_test)

    plt_line.preview_line3d(fig)

    fig.show()  # deactivate for auto unit testing
    assert True
