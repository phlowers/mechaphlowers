# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""
Tests for PositionEngine — standalone geometry computation without Plotly.
"""

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.geometry.group_points import GroupPoints
from mechaphlowers.core.geometry.points import Points, SparsePoints
from mechaphlowers.entities.arrays import ObstacleArray

SUPPORTS_POINTS = Points(
    coords=np.array(
        [
            [[0.0, 0.0, 20.0], [0.0, 0.0, 50.0], [0.0, 10.0, 50.0]],
            [[500.0, 0.0, 70.0], [500.0, 0.0, 103.0], [500.0, 10.0, 103.0]],
            [[1000.0, 0.0, 20.0], [1000.0, 0.0, 53.0], [1000.0, 10.0, 53.0]],
            [[1500.0, 0.0, 20.0], [1500.0, 0.0, 50.0], [1500.0, 10.0, 50.0]],
        ]
    )
)


INSULATORS_POINTS = Points(
    coords=np.array(
        [
            [
                [0.0, 10.0, 50.0],
                [2.99783566, 10.0, 49.88606426],
            ],
            [
                [500.0, 10.0, 103.0],
                [500.0, 10.0, 100.0],
            ],
            [
                [1000.0, 10.0, 53.0],
                [1000.0, 10.0, 50.0],
            ],
            [
                [1500.0, 10.0, 50.0],
                [1497.02812657, 10.0, 49.59016064],
            ],
        ]
    )
)


SPANS_POINTS = Points(
    coords=np.array(
        [
            [
                [2.99783566, 10.0, 49.88606426],
                [102.39826853, 10.0, 49.98589594],
                [201.7987014, 10.0, 55.0286192],
                [301.19913426, 10.0, 65.02669271],
                [400.59956713, 10.0, 80.00481789],
                [500.0, 10.0, 100.0],
            ],
            [
                [500.0, 10.0, 100.0],
                [600.0, 10.0, 79.90761484],
                [700.0, 10.0, 64.89267249],
                [800.0, 10.0, 54.91762777],
                [900.0, 10.0, 49.95753788],
                [1000.0, 10.0, 50.0],
            ],
            [
                [1000.0, 10.0, 50.0],
                [1099.40562531, 10.0, 40.01904745],
                [1198.81125063, 10.0, 34.99389657],
                [1298.21687594, 10.0, 34.91213083],
                [1397.62250125, 10.0, 39.7735482],
                [1497.02812657, 10.0, 49.59016064],
            ],
        ]
    )
)

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

OBSTACLE_POINTS = SparsePoints.builder_from_obstacle_array(obs_array)


class TestGroupPointsMethods:
    @pytest.fixture
    def group_points(self):
        return GroupPoints(SPANS_POINTS, SUPPORTS_POINTS, INSULATORS_POINTS)

    @pytest.fixture
    def group_points_obstacles(self):
        return GroupPoints(
            SPANS_POINTS, SUPPORTS_POINTS, INSULATORS_POINTS, OBSTACLE_POINTS
        )

    def test_create_group_points(self, group_points: GroupPoints):
        assert group_points.obstacles is None

    def test_create_group_points_with_obstacles(
        self, group_points_obstacles: GroupPoints
    ):
        assert group_points_obstacles.obstacles is not None

    def test_change_frame_no_angle(self, group_points: GroupPoints):
        result = group_points.change_frame(
            line_angle=np.array([0, 0, 0, 0]), frame_index=2
        )
        np.testing.assert_allclose(
            result.supports.coords[2, 0], np.array([0, 0, 20])
        )

    def test_change_frame_with_angle(self, group_points: GroupPoints):
        result = group_points.change_frame(
            line_angle=np.array(np.deg2rad([5, 10, 15, 20])), frame_index=2
        )
        np.testing.assert_allclose(
            result.supports.coords[2, 0], np.array([0, 0, 20])
        )

    def test_change_frame_loop(self, group_points: GroupPoints):
        group_points_projected_0 = group_points.change_frame(
            line_angle=np.array(np.deg2rad([5, 10, 15, 20])), frame_index=0
        )
        group_points_projected_2 = group_points_projected_0.change_frame(
            line_angle=np.array(np.deg2rad([5, 10, 15, 20])), frame_index=2
        )
        result = group_points_projected_2.change_frame(
            line_angle=np.array(np.deg2rad([5, 10, 15, 20])), frame_index=0
        )
        np.testing.assert_allclose(
            result.supports.coords, group_points_projected_0.supports.coords
        )
