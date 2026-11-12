# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from copy import deepcopy

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
        virtual_supports_list: list[int] | None = None,
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
        self.virtual_supports_list = (
            virtual_supports_list if virtual_supports_list is not None else []
        )

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

    def all_points_virtual_separated(self) -> dict[str, PointsT]:
        """Same than all_points, but spans with virtual supports are merged

        {
            "spans": self.spans,
            "supports": self.supports,
            "insulators": self.insulators,
            "obstacles": self.obstacles,
            "additional_points": self.additional_points,
            "virtual_supports": virtual_supports,
            "virtual_insulators": virtual_insulators,
        }
        """
        result_dict = deepcopy(self.all_points)  # type: ignore[arg-type]
        if len(self.virtual_supports_list) > 0:
            # careful to indices
            spans_coords = result_dict["spans"].coords
            # virtual_supports_array are the indices of the virtual supports BEFORE adding them
            # true indices are the position of the virtual supports AFTER adding them
            true_indices_vs = (
                np.array(sorted(self.virtual_supports_list), dtype=int)
                + np.array(range(len(self.virtual_supports_list)))
                + 1
            )
            for span_index in true_indices_vs:
                extracted_spans = np.take(
                    spans_coords, [span_index - 1, span_index], axis=0
                )
                new_span = extracted_spans.reshape(
                    1, extracted_spans.shape[1] * 2, 3
                )
                # Take one value every two points, for adjusting array length
                # The point of the virtual support will always be taken, because this point is duplicated
                new_span_resized = new_span[0][::2]
                # But we need the last point because it is the attachement point
                new_span_resized[-1] = new_span[0][-1]
                spans_coords[span_index - 1] = new_span_resized
                # add new span, and remove the useless one
            modified_spans_coords = np.delete(
                spans_coords, true_indices_vs, axis=0
            )
            result_dict["spans"] = Points(modified_spans_coords)  # type: ignore[assignment]

            supports_coords = result_dict["supports"].coords
            deleted_supports_coords = np.delete(
                supports_coords, true_indices_vs, axis=0
            )
            result_dict["supports"] = Points(deleted_supports_coords)  # type: ignore[assignment]
            virtual_supports_coords = supports_coords[true_indices_vs]
            result_dict["virtual_supports"] = Points(virtual_supports_coords)  # type: ignore[assignment]

            insulators_coords = result_dict["insulators"].coords
            deleted_insulators_coords = np.delete(
                insulators_coords, true_indices_vs, axis=0
            )
            result_dict["insulators"] = Points(deleted_insulators_coords)  # type: ignore[assignment]
            virtual_insulators_coords = insulators_coords[true_indices_vs]
            result_dict["virtual_insulators"] = Points(  # type: ignore[assignment]
                virtual_insulators_coords
            )

        return result_dict  # type: ignore[return-value]

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
        if self.obstacles is None:
            return {}
        return self.obstacles.dict_coords()

    def additional_points_dict(self):
        """Return additional_points as dict keyed by name."""
        if self.additional_points is None:
            return {}
        return self.additional_points.dict_coords()
