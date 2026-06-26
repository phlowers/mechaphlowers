# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest  # type: ignore[import-untyped]

from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.entities.arrays import (
    CableArray,
    ObstacleArray,
    SectionArray,
)
from mechaphlowers.plotting.plot import PlotEngine
from test.conftest import show_figures


def test_plot_obstacles(balance_engine_angles: BalanceEngine):
    plt_engine = PlotEngine(balance_engine_angles)
    balance_engine_angles.solve_adjustment()
    balance_engine_angles.solve_change_state(new_temperature=15)

    obs_array = ObstacleArray(
        pd.DataFrame(
            {
                "name": ["obs_0", "obs_0", "obs_1", "obs_1", "obs_1", "obs_2"],
                "point_index": [0, 1, 0, 1, 2, 0],
                "span_index": [0, 0, 1, 1, 1, 1],
                "x": [
                    100.0,
                    200.0,
                    100.0,
                    200.0,
                    300.0,
                    200.0,
                ],
                "y": [0.0, 10.0, 0.0, 10.0, 10.0, -20.0],
                "z": [0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
                "object_type": [
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                ],
            }
        )
    )
    plt_engine.add_obstacle_array(obs_array)
    fig = go.Figure()
    obstacle_dict = plt_engine.obstacles_dict()

    assert list(obstacle_dict.keys()) == ['obs_0', 'obs_1', 'obs_2']

    plt_engine.preview_line3d(fig)

    if show_figures:
        fig.show()

    points_result = plt_engine.get_obstacles_points()
    expected_result = np.array(
        [
            [np.nan, np.nan, np.nan],
            [100.0, 0.0, 0.0],
            [200.0, 10.0, 0.0],
            [np.nan, np.nan, np.nan],
            [598.76883406, 15.6434465, 0.0],
            [695.97332347, 41.16377641, 0.0],
            [794.74215753, 56.80722292, 50.0],
            [np.nan, np.nan, np.nan],
            [700.66635742, 11.5331262, 0.0],
        ]
    )
    # Previous test result before bugfix
    # expected_result = np.array(
    #     [
    #         [np.nan, np.nan, np.nan],
    #         [100.0, 0.0, 0.0],
    #         [200.0, 10.0, 0.0],
    #         [np.nan, np.nan, np.nan],
    #         [598.76883406, 15.6434465, 0.0],
    #         [699.10201277, 41.16377641, 0.0],
    #         [797.87084683, 56.80722292, 50.0],
    #         [np.nan, np.nan, np.nan],
    #         [694.40897882, 11.5331262, 0.0],
    #     ]
    # )
    np.testing.assert_allclose(points_result, expected_result)


def test_plot_obstacles_2d(balance_engine_angles: BalanceEngine):
    plt_engine = PlotEngine(balance_engine_angles)
    balance_engine_angles.solve_adjustment()
    balance_engine_angles.solve_change_state(new_temperature=15)

    obs_array = ObstacleArray(
        pd.DataFrame(
            {
                "name": ["obs_0", "obs_0", "obs_1", "obs_1", "obs_1", "obs_2"],
                "point_index": [0, 1, 0, 1, 2, 0],
                "span_index": [0, 0, 1, 1, 1, 1],
                "x": [
                    100.0,
                    200.0,
                    100.0,
                    200.0,
                    300.0,
                    200.0,
                ],
                "y": [0.0, 10.0, 0.0, 10.0, 10.0, -20.0],
                "z": [0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
                "object_type": [
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                ],
            }
        )
    )
    plt_engine.add_obstacle_array(obs_array)
    obstacle_dict = plt_engine.obstacles_dict(project=True, frame_index=1)

    assert list(obstacle_dict.keys()) == ['obs_0', 'obs_1', 'obs_2']

    fig_line = go.Figure()
    plt_engine.preview_line2d(fig_line, "line", 1)

    fig_profile = go.Figure()
    plt_engine.preview_line2d(fig_profile, "profile", 1)

    if show_figures:
        fig_line.show()
        fig_profile.show()
    points_result = plt_engine.get_obstacles_points()
    expected_result = np.array(
        [
            [np.nan, np.nan, np.nan],
            [100.0, 0.0, 0.0],
            [200.0, 10.0, 0.0],
            [np.nan, np.nan, np.nan],
            [598.76883406, 15.6434465, 0.0],
            [695.97332347, 41.16377641, 0.0],
            [794.74215753, 56.80722292, 50.0],
            [np.nan, np.nan, np.nan],
            [700.66635742, 11.5331262, 0.0],
        ]
    )
    # Previous test result before bugfix
    # expected_result = np.array(
    #     [
    #         [np.nan, np.nan, np.nan],
    #         [100.0, 0.0, 0.0],
    #         [200.0, 10.0, 0.0],
    #         [np.nan, np.nan, np.nan],
    #         [598.76883406, 15.6434465, 0.0],
    #         [699.10201277, 41.16377641, 0.0],
    #         [797.87084683, 56.80722292, 50.0],
    #         [np.nan, np.nan, np.nan],
    #         [694.40897882, 11.5331262, 0.0],
    #     ]
    # )
    np.testing.assert_allclose(points_result, expected_result)


def test_plot_obstacles_2d_angle_first_support(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [50, 100, 50, 60],
                "crossarm_length": [10, 10, 10, 10],
                "line_angle": [50, 10, 20, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 500, 500, np.nan],
                "insulator_mass": [100.0, 50.0, 50.0, 100.0],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        ),
        sagging_parameter=2000,
        sagging_temperature=15,
    )
    section_array.add_units({"line_angle": "grad"})
    balance_engine = BalanceEngine(
        cable_array=cable_array_AM600, section_array=section_array
    )

    plt_engine = PlotEngine(balance_engine)
    balance_engine.solve_adjustment()
    balance_engine.solve_change_state(new_temperature=15)

    obs_array = ObstacleArray(
        pd.DataFrame(
            {
                "name": ["obs_0", "obs_0", "obs_1", "obs_1", "obs_1", "obs_2"],
                "point_index": [0, 1, 0, 1, 2, 0],
                "span_index": [0, 0, 1, 1, 1, 1],
                "x": [
                    100.0,
                    200.0,
                    100.0,
                    200.0,
                    300.0,
                    200.0,
                ],
                "y": [0.0, 10.0, 0.0, 10.0, 10.0, -20.0],
                "z": [0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
                "object_type": [
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                ],
            }
        )
    )
    plt_engine.add_obstacle_array(obs_array)
    obstacle_dict = plt_engine.obstacles_dict(project=True, frame_index=1)

    assert list(obstacle_dict.keys()) == ['obs_0', 'obs_1', 'obs_2']

    fig_line = go.Figure()
    plt_engine.preview_line2d(fig_line, "line", 1)

    fig_profile = go.Figure()
    plt_engine.preview_line2d(fig_profile, "profile", 1)

    if show_figures:
        fig_line.show()
        fig_profile.show()
    group_points = plt_engine.position_engine.get_group_points().change_frame(
        1
    )
    expected_result = np.array(
        [
            [-3.95075336e02, 6.25737860e01, 0.00000000e00],
            [-2.94742158e02, 5.68072229e01, 0.00000000e00],
            # we recognize input data here
            [100, 0, 0],
            [200, 10, 0],
            [300, 10, 50],
            [200, -20, 0],
        ]
    )
    np.testing.assert_allclose(
        group_points.obstacles.coords,  # type: ignore[union-attr]
        expected_result,
        atol=1e-8,
    )


def test_plot_obstacles_frame_index_out_of_range(
    balance_engine_angles: BalanceEngine,
):
    plt_engine = PlotEngine(balance_engine_angles)
    balance_engine_angles.solve_adjustment()
    balance_engine_angles.solve_change_state(new_temperature=15)

    obs_array = ObstacleArray(
        pd.DataFrame(
            {
                "name": ["obs_0", "obs_0", "obs_1", "obs_1", "obs_1", "obs_2"],
                "point_index": [0, 1, 0, 1, 2, 0],
                "span_index": [0, 0, 1, 1, 1, 1],
                "x": [
                    100.0,
                    200.0,
                    100.0,
                    200.0,
                    300.0,
                    200.0,
                ],
                "y": [0.0, 10.0, 0.0, 10.0, 10.0, -20.0],
                "z": [0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
                "object_type": [
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                ],
            }
        )
    )
    plt_engine.add_obstacle_array(obs_array)
    with pytest.raises(ValueError):
        plt_engine.obstacles_dict(project=True, frame_index=5)
    with pytest.raises(ValueError):
        plt_engine.obstacles_dict(project=True, frame_index=-1)
