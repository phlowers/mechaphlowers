# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from copy import deepcopy

import numpy as np

from mechaphlowers.config import options
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
        distances: dict[str, dict[int, DistanceResult]] | None = None,
    ):
        self.line_angle = line_angle
        self.spans = spans
        self.supports = supports
        self.insulators = insulators
        self.obstacles = obstacles
        # dictionary of DistanceResult (result of get_distances_from_obstacles)
        self.distances = distances
        self.current_frame = -1

    @property
    def all_points(self) -> dict[str, PointsT]:
        """Dict of Points objects. If an attribute is set to None, it is not included

        {
            "spans": self.spans,
            "supports": self.supports,
            "insulators": self.insulators,
            "obstacles": self.obstacles,
        }
        """
        result_dict = {}
        attributes_points = ["spans", "supports", "insulators", "obstacles"]
        for name, points in self.__dict__.items():
            if points is not None and name in attributes_points:
                result_dict[name] = points
        return result_dict

    def get_aspect_ratio(
        self,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        z_scale: float = 1.0,
    ):
        all_points = self._array_all_coords_flattened()
        if all_points.size == 0:
            raise ValueError(
                "At least one Points object must contain at least one point to compute aspect ratio"
            )

        # Extract x, y, z coordinates
        xs = all_points[:, 0]
        ys = all_points[:, 1]
        zs = all_points[:, 2]

        # Compute ranges using nanmin/nanmax to handle NaN values
        x_range = np.nanmax(xs) - np.nanmin(xs)
        y_range = np.nanmax(ys) - np.nanmin(ys)
        z_range = np.nanmax(zs) - np.nanmin(zs)

        # Handle edge case where all values in an axis are NaN
        if np.isnan(x_range) or np.isnan(y_range) or np.isnan(z_range):
            raise ValueError(
                "Cannot compute aspect ratio because at least one axis has only NaN values"
            )

        # Normalize by the maximum range
        max_range = max(x_range, y_range, z_range)
        if max_range == 0:
            raise ValueError(
                "Data has zero spatial extent; cannot compute aspect ratio"
            )

        # Compute normalized ranges and clamp zero-extent axes to a small epsilon
        norm_x = (
            x_range / max_range
            if x_range > 0
            else options.graphics.aspect_epsilon
        )
        norm_y = (
            y_range / max_range
            if y_range > 0
            else options.graphics.aspect_epsilon
        )
        norm_z = (
            z_range / max_range
            if z_range > 0
            else options.graphics.aspect_epsilon
        )

        aspect_x = norm_x * x_scale
        aspect_y = norm_y * y_scale
        aspect_z = norm_z * z_scale

        return {
            "x": float(aspect_x),
            "y": float(aspect_y),
            "z": float(aspect_z),
        }

    # TODO:
    def filter(self, start: int, end: int):
        pass

    def _array_all_coords_flattened(self) -> np.ndarray:
        """Returns an array containing all points flattened:

        [[x0, y0, z0],
        [x1, y1, z1],
        [x2, y2, z2],
        ...
        ]

        Supports, spans, insulators and obstacles included
        """
        array_points = [
            points_object.points()
            for points_object in self.all_points.values()
        ]
        result = np.concatenate(array_points)
        return result

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
        if not isinstance(self.supports, Points):
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
    ):
        """Projection for dict of DistanceResult

        Directly modify objects

        Args:
            translation_vector (np.ndarray): _description_
            angle_to_project (np.float64): _description_

        Raises:
            TypeError: _description_

        Returns:
            _type_: _description_
        """
        # loop on self.distances (dict) and operate and change frame on each DistanceReuslt
        if not isinstance(self.distances, dict):
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
            translation_vector (np.ndarray): _description_
            angle_to_project (np.float64): _description_

        Returns:
            dict: _description_
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

    def __copy__(self) -> GroupPoints:
        return GroupPoints(
            self.line_angle,
            self.spans,
            self.supports,
            self.insulators,
            self.obstacles,
            self.distances,
        )
