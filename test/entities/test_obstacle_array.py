# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mechaphlowers.entities.arrays import (
    ObstacleArray,
)


def test_create_obstacle_array() -> None:
    input_data = {
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
        "y": [0.0, 10.0, 0.0, 0.0, 10.0, 0.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "object_type": [
            "ground",
            "ground",
            "ground",
            "ground",
            "ground",
            "ground",
        ],
    }
    ObstacleArray(pd.DataFrame(input_data))


def test_sort_obstacle_array() -> None:
    input_data = {
        "name": ["obs_0", "obs_1", "obs_0", "obs_2", "obs_1", "obs_1"],
        "point_index": [0, 1, 1, 0, 2, 0],
        "span_index": [0, 1, 0, 1, 1, 1],
        "x": [
            100.0,
            200.0,
            100.0,
            200.0,
            300.0,
            200.0,
        ],
        "y": [0.0, 10.0, 0.0, 0.0, 10.0, 0.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "object_type": [
            "ground",
            "ground",
            "ground",
            "ground",
            "ground",
            "ground",
        ],
    }
    obs_array = ObstacleArray(pd.DataFrame(input_data))
    assert obs_array.data["name"].to_list() == [
        'obs_0',
        'obs_0',
        'obs_1',
        'obs_1',
        'obs_1',
        'obs_2',
    ]
    assert obs_array.data["point_index"].to_list() == [0, 1, 0, 1, 2, 0]


def test_obstacle_array_duplicate_point() -> None:
    input_data = {
        "name": ["obs_0", "obs_0", "obs_1", "obs_1", "obs_1", "obs_2"],
        "point_index": [0, 1, 0, 1, 1, 0],
        "span_index": [0, 0, 1, 1, 1, 1],
        "x": [
            100.0,
            200.0,
            100.0,
            200.0,
            300.0,
            200.0,
        ],
        "y": [0.0, 10.0, 0.0, 0.0, 10.0, 0.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "object_type": [
            "ground",
            "ground",
            "ground",
            "ground",
            "ground",
            "ground",
        ],
    }
    with pytest.raises(ValueError):
        # should return error because two points of the same obstacle have the same index
        ObstacleArray(pd.DataFrame(input_data))


def test_obstacle_array_different_span() -> None:
    input_data = {
        "name": ["obs_0", "obs_0", "obs_1", "obs_1", "obs_1", "obs_2"],
        "point_index": [0, 1, 0, 1, 2, 0],
        "span_index": [0, 0, 1, 0, 1, 1],
        "x": [
            100.0,
            200.0,
            100.0,
            200.0,
            300.0,
            200.0,
        ],
        "y": [0.0, 10.0, 0.0, 0.0, 10.0, 0.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "object_type": [
            "ground",
            "ground",
            "ground",
            "ground",
            "ground",
            "ground",
        ],
    }
    with pytest.raises(ValueError):
        # should return error because two points of the same obstacle have different span_index
        ObstacleArray(pd.DataFrame(input_data))


def test_add_obstacle() -> None:
    input_data = {
        "name": ["obs_0", "obs_0"],
        "point_index": [0, 1],
        "span_index": [0, 0],
        "x": [
            100.0,
            200.0,
        ],
        "y": [0.0, 10.0],
        "z": [0.0, 0.0],
        "object_type": [
            "ground",
            "ground",
        ],
    }
    obstacle_array = ObstacleArray(pd.DataFrame(input_data))
    obstacle_array.add_obstacle(
        name="obs_1",
        span_index=1,
        coords=np.array([[50, 0, 0], [100, 0, 10], [150, 10, 0], [200, 0, 0]]),
        support_reference='left',
    )
    obstacle_array.add_obstacle(
        name="obs_2",
        span_index=1,
        coords=np.array([[35, 0, 0], [100, 0, 10]]),
        support_reference='right',
        span_length=np.array([500, 400, 450, np.nan]),
    )

    expected_df = pd.DataFrame(
        {
            "name": [
                "obs_0",
                "obs_0",
                "obs_1",
                "obs_1",
                "obs_1",
                "obs_1",
                "obs_2",
                "obs_2",
            ],
            "point_index": [0, 1, 0, 1, 2, 3, 0, 1],
            "span_index": [0, 0, 1, 1, 1, 1, 1, 1],
            "x": [100.0, 200.0, 50.0, 100.0, 150.0, 200.0, 365.0, 300.0],
            "y": [0.0, 10.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            "object_type": [
                "ground",
                "ground",
                "ground",
                "ground",
                "ground",
                "ground",
                "ground",
                "ground",
            ],
        }
    )

    assert_frame_equal(obstacle_array.data, expected_df, check_like=True)


def test_add_obstacle_bad_arguments() -> None:
    input_data = {
        "name": ["obs_0", "obs_0"],
        "point_index": [0, 1],
        "span_index": [0, 0],
        "x": [
            100.0,
            200.0,
        ],
        "y": [0.0, 10.0],
        "z": [0.0, 0.0],
        "object_type": [
            "ground",
            "ground",
        ],
    }
    obstacle_array = ObstacleArray(pd.DataFrame(input_data))
    # coords bad shape
    with pytest.raises(TypeError):
        obstacle_array.add_obstacle(
            name="obs_1",
            span_index=1,
            coords=np.array([50, 0, 0]),
            support_reference='left',
        )
    with pytest.raises(TypeError):
        obstacle_array.add_obstacle(
            name="obs_1",
            span_index=1,
            coords=np.array([[50, 0], [60, 0]]),
            support_reference='left',
        )
    # span_length missing
    with pytest.raises(TypeError):
        obstacle_array.add_obstacle(
            name="obs_2",
            span_index=1,
            coords=np.array([[35, 0, 0], [100, 0, 10]]),
            support_reference='right',
        )
