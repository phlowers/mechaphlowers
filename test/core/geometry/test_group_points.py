# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from copy import copy

import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.geometry.distances import (
    PointDistanceResult,
    ClosestDistanceResult,
    ObstacleDistances,
)
from mechaphlowers.core.geometry.group_points import GroupPoints
from mechaphlowers.core.geometry.points import Points, SparsePoints
from mechaphlowers.entities.arrays import ObstacleArray


@pytest.fixture
def supports_points() -> Points:
    return Points(
        coords=np.array(
            [
                [[0.0, 0.0, 20.0], [0.0, 0.0, 50.0], [0.0, 10.0, 50.0]],
                [
                    [500.0, 0.0, 70.0],
                    [500.0, 0.0, 103.0],
                    [500.0, 10.0, 103.0],
                ],
                [
                    [1000.0, 0.0, 20.0],
                    [1000.0, 0.0, 53.0],
                    [1000.0, 10.0, 53.0],
                ],
                [
                    [1500.0, 0.0, 20.0],
                    [1500.0, 0.0, 50.0],
                    [1500.0, 10.0, 50.0],
                ],
            ]
        )
    )


@pytest.fixture
def insulators_points() -> Points:
    return Points(
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


@pytest.fixture
def spans_points() -> Points:
    return Points(
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


@pytest.fixture
def obs_array() -> ObstacleArray:
    return ObstacleArray(
        pd.DataFrame(
            {
                "name": [
                    "obs_0",
                    "obs_0",
                    "obs_1",
                    "obs_1",
                    "obs_1",
                    "obs_2",
                ],
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


@pytest.fixture
def obstacle_points(obs_array: ObstacleArray) -> SparsePoints:
    return SparsePoints.builder_from_obstacle_array(obs_array)


@pytest.fixture
def distance_result() -> PointDistanceResult:
    return PointDistanceResult(
        np.array([100.0000, 6.7000, 0.0000]),
        np.array([100.0000, 9.9940, 49.9035]),
        np.array([-0.0000, 1.0000, 0.0000]),
        np.array([0.0000, -0.0000, 1.0000]),
        50.894348,
        9.993974,
        49.903458,
    )


@pytest.fixture
def distance_dict(
    distance_result: PointDistanceResult,
) -> dict[str, ObstacleDistances]:
    def points(nb_points: int) -> dict[int, PointDistanceResult]:
        return {i: copy(distance_result) for i in range(nb_points)}

    return {
        "obs_0": ObstacleDistances("obs_0", 0, "points", points(2), None),
        "obs_1": ObstacleDistances(
            "obs_1",
            1,
            "polyline",
            points(3),
            ClosestDistanceResult("obs_1", 1, copy(distance_result), 0, 0.5),
        ),
        "obs_2": ObstacleDistances("obs_2", 1, "points", points(1), None),
    }


@pytest.fixture
def group_points(
    spans_points: Points,
    supports_points: Points,
    insulators_points: Points,
) -> GroupPoints:
    return GroupPoints(
        np.array([0, 0, 0, 0]),
        spans_points,
        supports_points,
        insulators_points,
    )


@pytest.fixture
def group_points_obstacles(
    spans_points: Points,
    supports_points: Points,
    insulators_points: Points,
    obstacle_points: SparsePoints,
) -> GroupPoints:
    return GroupPoints(
        np.array([0, 0, 0, 0]),
        spans_points,
        supports_points,
        insulators_points,
        obstacle_points,
    )


@pytest.fixture
def additional_points(obs_array: ObstacleArray) -> SparsePoints:
    return SparsePoints.builder_from_obstacle_array(obs_array)


@pytest.fixture
def group_points_distances(
    spans_points: Points,
    supports_points: Points,
    insulators_points: Points,
    obstacle_points: SparsePoints,
    distance_dict: dict[str, ObstacleDistances],
) -> GroupPoints:
    return GroupPoints(
        np.array([0, 0, 0, 0]),
        spans_points,
        supports_points,
        insulators_points,
        obstacle_points,
        distances=distance_dict,
    )


@pytest.fixture
def group_points_additional(
    spans_points: Points,
    supports_points: Points,
    insulators_points: Points,
    additional_points: SparsePoints,
) -> GroupPoints:
    return GroupPoints(
        np.array([0, 0, 0, 0]),
        spans_points,
        supports_points,
        insulators_points,
        additional_points=additional_points,
    )


class TestMethods:
    def test_create_group_points(self, group_points: GroupPoints):
        assert group_points.obstacles is None

    def test_create_group_points_with_obstacles(
        self, group_points_obstacles: GroupPoints
    ):
        assert group_points_obstacles.obstacles is not None

    def test_aspect_ratio(self, group_points_obstacles: GroupPoints):
        group_points_obstacles.get_aspect_ratio()

    def test_create_group_points_with_additional_points(
        self, group_points_additional: GroupPoints
    ):
        assert group_points_additional.additional_points is not None

    def test_additional_points_in_all_points(
        self, group_points_additional: GroupPoints
    ):
        assert "additional_points" in group_points_additional.all_points

    def test_additional_points_dict(
        self, group_points_additional: GroupPoints
    ):
        result = group_points_additional.additional_points_dict()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_additional_points_dict_empty_when_none(
        self, group_points: GroupPoints
    ):
        assert group_points.additional_points_dict() == {}

    def test_change_frame_propagates_to_additional_points(
        self, group_points_additional: GroupPoints
    ):
        original_x = group_points_additional.additional_points.x.copy()  # type: ignore[union-attr]
        result = group_points_additional.change_frame(frame_index=2)
        # frame change translates by support[2] offset — x must differ from original
        assert not np.allclose(
            result.additional_points.x,  # type: ignore[union-attr]
            original_x,
        )


class TestChangeFrame:
    def test_change_frame_no_angle(self, group_points: GroupPoints):
        result = group_points.change_frame(frame_index=2)
        np.testing.assert_allclose(
            result.supports.coords[2, 0],  # type: ignore[union-attr]
            np.array([0, 0, 20]),
        )

    def test_change_frame_with_angle(self, group_points: GroupPoints):
        group_points.line_angle = np.array(np.deg2rad([5, 10, 15, 20]))
        result = group_points.change_frame(frame_index=2)
        np.testing.assert_allclose(
            result.supports.coords[2, 0],  # type: ignore[union-attr]
            np.array([0, 0, 20]),
        )

    def test_change_frame_loop(self, group_points: GroupPoints):
        group_points.line_angle = np.array(np.deg2rad([0, 10, 15, 20]))

        group_points_projected_2 = group_points.change_frame(frame_index=2)
        result = group_points_projected_2.change_frame(frame_index=0)
        np.testing.assert_allclose(
            result.supports.coords,  # type: ignore[union-attr]
            group_points.supports.coords,  # type: ignore[union-attr]
            atol=1e-10,
        )

    def test_change_frame_loop__angle_first_support(
        self, group_points: GroupPoints
    ):
        group_points.line_angle = np.array(np.deg2rad([5, 10, 15, 20]))

        group_points_projected_0 = group_points.change_frame(frame_index=0)
        group_points_projected_2 = group_points_projected_0.change_frame(
            frame_index=2
        )
        result = group_points_projected_2.change_frame(frame_index=0)
        np.testing.assert_allclose(
            result.supports.coords,  # type: ignore[union-attr]
            group_points_projected_0.supports.coords,  # type: ignore[union-attr]
        )

    def test_change_frame_loop__minus_one(self, group_points: GroupPoints):
        group_points.line_angle = np.array(np.deg2rad([5, 10, 15, 20]))

        group_points_projected_2 = group_points.change_frame(frame_index=2)
        result = group_points_projected_2.change_frame(frame_index=-1)
        np.testing.assert_allclose(
            result.supports.coords,  # type: ignore[union-attr]
            group_points.supports.coords,  # type: ignore[union-attr]
            atol=1e-10,
        )

    def test_change_frame_original_unchanged(
        self, group_points_distances: GroupPoints
    ):
        group_points_distances.line_angle = np.array(
            np.deg2rad([5, 10, 15, 20])
        )
        group_points_distances.change_frame(frame_index=2)
        expected_supports_unchanged = np.array(
            [
                [[0.0, 0.0, 20.0], [0.0, 0.0, 50.0], [0.0, 10.0, 50.0]],
                [
                    [500.0, 0.0, 70.0],
                    [500.0, 0.0, 103.0],
                    [500.0, 10.0, 103.0],
                ],
                [
                    [1000.0, 0.0, 20.0],
                    [1000.0, 0.0, 53.0],
                    [1000.0, 10.0, 53.0],
                ],
                [
                    [1500.0, 0.0, 20.0],
                    [1500.0, 0.0, 50.0],
                    [1500.0, 10.0, 50.0],
                ],
            ]
        )
        np.testing.assert_equal(
            group_points_distances.supports.coords,  # type: ignore[union-attr]
            expected_supports_unchanged,
        )
        np.testing.assert_equal(
            group_points_distances.distances["obs_1"].points[0].point_base,  # type: ignore[index]
            np.array([100.0000, 6.7000, 0.0000]),
        )


class TestGroupDistances:
    def test_distances_no_angle(self, group_points_distances: GroupPoints):
        group_points_distances._change_frame_distances(
            translation_vector=np.array([500, 0, 0]),
            angle_to_project=np.float64(0.0),
        )
        group_points_distances.distances
        assert True

    def test_change_frame_moves_points_and_minimum(
        self, group_points_distances: GroupPoints
    ):
        moved = group_points_distances.change_frame(frame_index=1)
        expected_base = np.array([100.0, 6.7, 0.0]) - np.array([500, 0, 0])

        obs_1 = moved.distances["obs_1"]  # type: ignore[index]
        np.testing.assert_allclose(obs_1.points[0].point_base, expected_base)
        np.testing.assert_allclose(
            obs_1.minimum.closest_obstacle_point,  # type: ignore[union-attr]
            expected_base,
        )
        # original object is unchanged
        np.testing.assert_allclose(
            group_points_distances.distances[
                "obs_1"
            ].minimum.closest_obstacle_point,  # type: ignore[index, union-attr]
            np.array([100.0, 6.7, 0.0]),
        )

    def test_distances_with_angle(self, group_points_distances: GroupPoints):
        group_points_distances.line_angle = np.array(
            np.deg2rad([5, 10, 15, 20])
        )
        group_points_distances._change_frame_distances(
            translation_vector=np.array([500, 0, 0]),
            angle_to_project=np.deg2rad(15),
        )
        group_points_distances.distances
        assert True


class TestVirtualSupports:
    def test_virtual_support(self, group_points: GroupPoints):
        group_points.virtual_supports_list = [1]
        all_points = group_points.all_points_virtual_separated()
        spans = all_points["spans"]
        supports = all_points["supports"]
        insulators = all_points["insulators"]
        assert len(spans.coords) == 2
        assert len(supports.coords) == 3
        assert len(insulators.coords) == 3
        first_point = [500.0, 10.0, 100.0]
        middle_point = [1000.0, 10.0, 50.0]
        last_point = [1497.02812657, 10.0, 49.59016064]

        assert first_point in spans.coords[1].tolist()
        assert middle_point in spans.coords[1].tolist()
        assert last_point in spans.coords[1].tolist()
