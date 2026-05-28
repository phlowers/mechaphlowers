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
        distances: dict[str, dict[int, DistanceResult]]
        | None = None,  # dictionary of DistanceResult (result of get_distances_from_obstacles)
    ):
        self.line_angle = line_angle
        self.spans = spans
        self.supports = supports
        self.insulators = insulators
        self.obstacles = obstacles
        self.distances = distances
        self.current_frame = -1

    # useless? use __dict__ instead?
    # pros: we can control the attributes we include in all_points
    @property
    def all_points(self) -> dict[str, PointsT]:
        """List of Points objects. Does not contain attributes if set to None"""
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

    def _array_all_coords_flattened(self):
        array_points = [
            points_object.points()
            for points_object in self.all_points.values()
        ]
        result = np.concatenate(array_points)
        return result

    # add inplace?
    def change_frame(self, frame_index=0) -> GroupPoints:
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

        translation_vector = -self.supports.coords[frame_index, 0]
        # set z coordinate to zero: only tanslate horizontally
        translation_vector[2] = 0.0

        new_group_points = deepcopy(self)

        new_group_points.project_coords(translation_vector, angle_to_project)
        if new_group_points.distances is not None:
            new_group_points._change_frame_distances(
                translation_vector, angle_to_project
            )
        new_group_points.current_frame = frame_index
        return new_group_points

    def _change_frame_distances(
        self, translation_vector: np.ndarray, angle_to_project: np.float64
    ):
        # called by change_frame
        # loop on self.distances (dict) and operate and change frame on each DistanceReuslt
        if not isinstance(self.distances, dict):
            raise TypeError("self.distances need to be a dictionary")

        for obstacle_name, obstacle_dict in self.distances.items():
            for point_index, distance_result in obstacle_dict.items():
                obstacle_dict[point_index] = distance_result.change_frame(
                    translation_vector, angle_to_project
                )
        return self.distances

    def project_coords(
        self, translation_vector: np.ndarray, angle_to_project: np.float64
    ) -> dict:
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
