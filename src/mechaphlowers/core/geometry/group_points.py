# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from mechaphlowers.core.geometry.distances import DistanceResult
from mechaphlowers.core.geometry.points import (
    Points,
    PointsT,
    SparsePoints,
    compute_new_frame,
)


class GroupPoints:
    def __init__(
        self,
        line_angle: np.ndarray,
        spans: Points | None = None,
        supports: Points | None = None,
        insulators: Points | None = None,
        obstacles: SparsePoints | None = None,
        additional_points: SparsePoints | None = None,
        distances: dict[str, dict[int, DistanceResult]] | None = None,
    ):
        self.line_angle = line_angle
        self.spans = spans
        self.supports = supports
        self.insulators = insulators
        self.obstacles = obstacles
        self.additional_points = additional_points
        # dictionary of DistanceResult (result of get_distances_from_obstacles)
        self.distances = distances
        self.current_frame = -1

    @property
    def all_points(self) -> dict[str, PointsT]:
        """Dict of Points objects. If an attribute is set to None, it is not included
        Distances are NOT included in this dict

        {
            "spans": self.spans,
            "supports": self.supports,
            "insulators": self.insulators,
            "obstacles": self.obstacles,
            "additional_points": self.additional_points,
        }
        """
        result_dict = {}
        attributes_points = [
            "spans",
            "supports",
            "insulators",
            "obstacles",
            "additional_points",
        ]
        for name, points in self.__dict__.items():
            if points is not None and name in attributes_points:
                result_dict[name] = points
        return result_dict

    def get_all_objects_dict(self, reversed_y_axis=False) -> dict:
        """Dict of all objects. If an attribute is set to None, it is not included
        Distances are included in this dict

        {
            "spans": self.spans,
            "supports": self.supports,
            "insulators": self.insulators,
            "obstacles": self.obstacles,
            "additional_points": self.additional_points,
            "distances": self.distances,
        }

        Option to reverse the y axis for displaying the line in 2D (may be more intuitive)

        Args:
            reversed_y_axis (bool): If True: reverse the y axis. Defaults to False
        """
        if reversed_y_axis:
            result_dict: dict[str, Any] = {}
            points_objects = ["spans", "supports", "insulators"]
            for name, points in self.__dict__.items():
                if points is not None and name in points_objects:
                    x, y, z = points.vectors
                    inverted_points = Points.from_vectors(x, -y, z)
                    result_dict[name] = inverted_points
            if isinstance(self.obstacles, SparsePoints):
                reversed_obstacle = deepcopy(self.obstacles)
                reversed_obstacle.y = -reversed_obstacle.y
                result_dict["obstacles"] = reversed_obstacle
            if isinstance(self.additional_points, SparsePoints):
                reversed_additional = deepcopy(self.additional_points)
                reversed_additional.y = -reversed_additional.y
                result_dict["additional_points"] = reversed_additional
            if self.distances is not None:
                reversed_distances_dict = self._reverse_y_axis_distances(
                    deepcopy(self.distances)
                )
                result_dict["distances"] = reversed_distances_dict
        else:
            result_dict = self.all_points
            if isinstance(self.distances, dict):
                result_dict["distances"] = self.distances  # type:ignore[assignment]
        return result_dict

    def _reverse_y_axis_distances(self, distances_dict: dict) -> dict:
        for obstacle_name, obstacle_dict in distances_dict.items():
            for point_index, distance_result in obstacle_dict.items():
                distances_dict[obstacle_name][point_index] = (
                    distance_result.generate_with_reversed_y_axis()
                )

        return distances_dict

    def get_aspect_ratio(
        self,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        z_scale: float = 1.0,
    ):
        from mechaphlowers.plotting.utils import compute_aspect_ratio

        return compute_aspect_ratio(
            *self.all_points.values(),
            x_scale=x_scale,
            y_scale=y_scale,
            z_scale=z_scale,
        )

    # add inplace argument?
    def change_frame(self, frame_index=-1) -> GroupPoints:
        """Change frame of all objects: supports, insulators, spans, obstacles and distances.

        Returns a new object, keep the current one unchanged.

        frame_index can be set to -1 to get initial frame: origin to first support, without taking into account first angle.

        frame_index set to 0 will take into account the first angle.

        Args:
            frame_index (int, optional): index of the frame. Defaults to -1.

        Raises:
            TypeError: if self.supports is not initialized
            ValueError: if frame_index is out of range [-1, nb_supports - 1]

        Returns:
            GroupPoints: new object GroupPoints projected in the selected frame
        """
        if self.supports is None:
            raise TypeError(
                "attribute self.support need to be a Points object"
            )
        if frame_index > self.current_frame:
            angle_to_project = np.sum(
                self.line_angle[self.current_frame + 1 : frame_index + 1]
            )
        elif frame_index < self.current_frame:
            angle_to_project = -np.sum(
                self.line_angle[frame_index + 1 : self.current_frame + 1]
            )

        else:
            return deepcopy(self)

        if frame_index == -1:
            translation_vector = -self.supports.coords[0, 0]
        elif 0 <= frame_index < len(self.line_angle):
            translation_vector = -self.supports.coords[frame_index, 0]
        else:
            raise ValueError(
                f"frame_index should be between -1 and {len(self.line_angle) - 1}. Received {frame_index}"
            )
        # set z coordinate to zero: only translate horizontally
        translation_vector[2] = 0.0

        new_group_points = deepcopy(self)

        new_group_points._change_frame_points(
            translation_vector, angle_to_project
        )
        if new_group_points.distances is not None:
            new_group_points._change_frame_distances(
                translation_vector, angle_to_project
            )
        new_group_points.current_frame = frame_index
        return new_group_points

    def _change_frame_distances(
        self, translation_vector: np.ndarray, angle_to_project: np.float64
    ) -> dict:
        """Projection for dict of DistanceResult

        Directly modify objects

        Args:
            translation_vector (np.ndarray): translation vector
            angle_to_project (np.float64): angle of rotation (radians, anti-clockwise)

        Raises:
            TypeError: if self.distances does not exist

        Returns:
            dict: dictionnary of distances: same format as self.distances
        """
        # loop on self.distances (dict) and operate and change frame on each DistanceReuslt
        if self.distances is None:
            raise TypeError("self.distances need to be a dictionary")

        for obstacle_dict in self.distances.values():
            for point_index, distance_result in obstacle_dict.items():
                obstacle_dict[point_index] = distance_result.compute_new_frame(
                    translation_vector, angle_to_project
                )
        # return unused
        return self.distances

    def _change_frame_points(
        self, translation_vector: np.ndarray, angle_to_project: np.float64
    ) -> dict:
        """Projection for Points and SparsePoints objects.

        Directly modify objects

        Args:
            translation_vector (np.ndarray): translation vector
            angle_to_project (np.float64): angle of rotation (radians, anti-clockwise)

        Returns:
            dict: same format as self.all_points: contains spans/insulators/supports/obstacles if they exists
        """
        # modify current object
        result_points = {}
        for name, original_points in self.all_points.items():
            new_points = compute_new_frame(
                original_points, translation_vector, angle_to_project
            )
            result_points[name] = new_points
        return result_points

    def obstacle_dict(self):
        return self.obstacles.dict_coords()

    def additional_points_dict(self):
        """Return additional_points as dict keyed by name."""
        if self.additional_points is None:
            return {}
        return self.additional_points.dict_coords()
