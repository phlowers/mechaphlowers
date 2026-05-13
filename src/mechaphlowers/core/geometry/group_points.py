# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np

from mechaphlowers.core.geometry.points import (
    Points,
    SparsePoints,
    compute_new_frame,
)


class GroupPoints:
    def __init__(
        self,
        spans: Points | None = None,
        supports: Points | None = None,
        insulators: Points | None = None,
        obstacles: SparsePoints | None = None,
    ):
        self.spans = spans
        self.supports = supports
        self.insulators = insulators
        self.obstacles = obstacles

    # TODO: think about what to do for distances

    # useless? use __dict__ instead?
    # pros: we can control the attributes we include in all_points
    @property
    def all_points(self):
        """List of Points objects. Does not contain attributes if set to None"""
        result_dict = {}
        attributes_points = ["spans", "supports", "insulators", "obstacles"]
        for name, points in self.__dict__.items():
            if points is not None and name in attributes_points:
                result_dict[name] = points
        return result_dict

    # add inplace?
    # line_angle: method argument or class attribute?
    def change_frame(self, line_angle: np.ndarray, frame_index=0):
        if not isinstance(self.supports, Points):
            raise TypeError(
                "attribute self.support need to be a Points object"
            )
        translation_vector = -self.supports.coords[frame_index, 0]
        # set z coordinate to zero
        translation_vector[2] = 0.0
        angle_to_project = np.cumsum(line_angle)[frame_index]
        result_group_points = self.project_coords(
            translation_vector, angle_to_project
        )
        return result_group_points

    def project_coords(
        self, translation_vector: np.ndarray, angle_to_project: np.float64
    ):
        result_points = {}
        for name, original_points in self.all_points.items():
            new_points = compute_new_frame(
                original_points, translation_vector, angle_to_project
            )
            result_points[name] = new_points
        return GroupPoints(**result_points)
