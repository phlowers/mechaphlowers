# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]

from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.entities.arrays import ObstacleArray
from mechaphlowers.plotting.plot import PlotEngine


def test_plot_obstacles(balance_engine_angles: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(balance_engine_angles)
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
    plt_engine.add_obstacles(obs_array)
    fig = go.Figure()
    # plt_engine.section_pts.get_obstacle_coords()
    # plt_engine.section_pts.obstacles_points.dict_coords()
    plt_engine.preview_line3d(fig)
    # fig.show()
    points_result = plt_engine.get_obstacles_points()
    expected_result = np.array(
        [
            [np.nan, np.nan, np.nan],
            [100.0, 0.0, 0.0],
            [200.0, 10.0, 0.0],
            [np.nan, np.nan, np.nan],
            [598.76883406, 15.6434465, 0.0],
            [699.10201277, 41.16377641, 0.0],
            [797.87084683, 56.80722292, 50.0],
            [np.nan, np.nan, np.nan],
            [694.40897882, 11.5331262, 0.0],
        ]
    )
    np.testing.assert_allclose(points_result, expected_result)

    # TODO: test keys in dict
    plt_engine.section_pts.obstacles_points.dict_coords()
