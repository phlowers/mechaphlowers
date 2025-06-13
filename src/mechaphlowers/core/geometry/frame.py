# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.geometry.rotation import (
    rotation_quaternion,
    rotation_quaternion_same_axis,
)


class Frame:
    def __init__(
        self,
        origin: np.ndarray = np.full((4, 3), np.zeros(3)),
        axis: np.ndarray = np.full((4, 3, 3), np.eye(3)),
    ):
        self.origin: np.ndarray = origin
        #    [[x_0, y_0, z_0], #layer 0
        #     [x_1, y_1, z_1], #layer 1
        #     ...
        #    ]
        self.axis: np.ndarray = axis

    #    [[x_0, y_0, z_0], #layer 0
    #     [x_1, y_1, z_1], #layer 1
    #     ...
    #    ]

    @property
    def x_axis(self):
        return self.axis[:, 0]

    @property
    def y_axis(self):
        return self.axis[:, 1]

    @property
    def z_axis(self):
        return self.axis[:, 2]

    @property
    def angle_rotation(self):
        # result in degrees
        trace = np.trace(self.axis, axis1=1, axis2=2)
        theta = np.arccos((trace - 1) / 2)
        return np.rad2deg(theta)

    @property
    def rotation_axis(self) -> np.ndarray:
        # cf wikipedia
        # https://en.wikipedia.org/wiki/Rotation_matrix#Conversion from rotation matrix to axis–angle
        # Do better than this
        axis_reshaped = np.reshape(
            np.ravel(self.axis, order="F"), (9, self.axis.shape[0])
        )
        rotation_axis = np.array(
            [
                axis_reshaped[7] - axis_reshaped[5],
                axis_reshaped[2] - axis_reshaped[6],
                axis_reshaped[3] - axis_reshaped[1],
            ]
        )
        rotation_axis_reshaped = np.reshape(
            np.ravel(rotation_axis, order="F"), (self.axis.shape[0], 3)
        )
        simple_truth_table: np.ndarray = np.all(
            rotation_axis_reshaped == 0, axis=1
        )
        is_zero = np.repeat(simple_truth_table[:, np.newaxis], 3, axis=1)
        default_z_axis = np.array([[0, 0, 1]]).repeat(
            self.axis.shape[0], axis=0
        )
        # creates warning because dividing by zero
        rotation_axis_normalized = np.where(
            is_zero,
            default_z_axis,
            rotation_axis_reshaped
            / np.linalg.norm(rotation_axis_reshaped, axis=1)[:, np.newaxis],
        )
        return rotation_axis_normalized

    def unflatten(self, flattened_points: np.ndarray):
        return flattened_points.reshape(self.axis.shape)

    def translate_current(self, translation_vector: np.ndarray):
        # translation_vector has coords in current frame
        # rotate translation vector to current frame, using the rotation axis and angle from absolute frame
        translation_vector_current_frame = rotation_quaternion(
            translation_vector, -self.angle_rotation, self.rotation_axis
        )
        self.origin = self.origin + translation_vector_current_frame

    def translate_absolute(self, translation_vector: np.ndarray):
        # translation_vector in absolute coords
        self.origin += translation_vector

    # frame rotation: around itself
    def rotate(self, line_angles_layer: np.ndarray, rotation_axes: np.ndarray):
        line_angles_points = np.repeat(
            line_angles_layer, self.axis.shape[1], axis=0
        )
        rotation_axes_array = np.repeat(
            rotation_axes, self.axis.shape[1], axis=0
        )
        flattened_points = self.axis.reshape(
            -1, 3
        )  # to refector later with flatten method
        coords_rotated = rotation_quaternion(
            flattened_points, line_angles_points, rotation_axes_array
        )
        self.axis = self.unflatten(coords_rotated)
        self.rotation_axis

    def rotate_same_axis(
        self, line_angles_layer: np.ndarray, rotation_axis: np.ndarray
    ):
        line_angle_array = np.repeat(
            line_angles_layer, self.axis.shape[1], axis=0
        )
        flattened_points = self.axis.reshape(-1, 3)
        coords_rotated = rotation_quaternion_same_axis(
            flattened_points, line_angle_array, rotation_axis
        )
        self.axis = self.unflatten(coords_rotated)


class Points:
    def __init__(self, coords: np.ndarray, frame: Frame = Frame()):
        self.coords: np.ndarray = coords.copy()  # give copy
        # [
        #     [   # layer 0
        #         [x0_0, y0_0, z0_0],
        #         [x1_0, y1_0, z1_0]
        #     ],
        #     [   # layer 1
        #         [x0_1, y0_1, z0_1],
        #         [x1_1, y1_1, z1_1]
        #     ],
        #     [   # layer 2
        #         [x0_2, y0_2, z0_2],
        #         [x1_2, y1_2, z1_2]
        #     ],
        # ]
        self.frame: Frame = frame  # copy ?
        # check if coords and frame are compatible

    def translate_all(self, translation_vector: np.ndarray):
        translation_vector_points = np.repeat(
            translation_vector[:, np.newaxis], self.coords.shape[1], axis=1
        )
        self.coords += translation_vector_points

    def translate_layer(self, translation_vector: np.ndarray, layer: int):
        self.coords[layer] += translation_vector

    def translate_point(
        self, translation_vector: np.ndarray, layer: int, index: tuple
    ):
        self.coords[layer, index] += translation_vector

    def flatten(self):
        return self._flatten(self.coords)

    @staticmethod
    def _flatten(points):
        return points.reshape(-1, 3)

    def unflatten(self, flattened_points: np.ndarray):
        return flattened_points.reshape(self.coords.shape)

    def rotate_all(
        self, line_angles_layer: np.ndarray, rotation_axes: np.ndarray
    ):
        line_angles_points = np.repeat(
            line_angles_layer, self.coords.shape[1], axis=0
        )
        rotation_axes_array = np.repeat(
            rotation_axes, self.coords.shape[1], axis=0
        )
        flattened_points = self.flatten()
        coords_rotated = rotation_quaternion(
            flattened_points, line_angles_points, rotation_axes_array
        )
        self.coords = self.unflatten(coords_rotated)

    def rotate_one_angle_per_point(
        self, line_angles_points: np.ndarray, rotation_axes: np.ndarray
    ):
        rotation_axes_array = np.repeat(
            rotation_axes, self.coords.shape[1], axis=0
        )
        flattened_points = self.flatten()
        coords_rotated = rotation_quaternion(
            flattened_points, line_angles_points, rotation_axes_array
        )
        self.coords = self.unflatten(coords_rotated)

    def rotate_same_axis(
        self, line_angles_layer: np.ndarray, rotation_axis: np.ndarray
    ):
        line_angle_array = np.repeat(
            line_angles_layer, self.coords.shape[1], axis=0
        )
        flattened_points = self.flatten()
        coords_rotated = rotation_quaternion_same_axis(
            flattened_points, line_angle_array, rotation_axis
        )
        self.coords = self.unflatten(coords_rotated)

    def nan_points(self):
        a, b, c = self.coords.shape
        return np.hstack([self.coords, np.nan * np.ones((a, 1, c))])

    def coords_for_plot(self):
        return self._flatten(self.nan_points())

    # transform -> unusable because of order?
    def transform(
        self,
        translation_vector: np.ndarray,
        line_angle: np.ndarray,
        rotation_axis: np.ndarray = np.array([0, 0, 1]),
    ):
        self.translate_all(translation_vector)
        self.rotate_all(line_angle, rotation_axis)

    def transform_frame(
        self,
        translation_vector: np.ndarray,
        line_angle: np.ndarray,
        rotation_axis: np.ndarray = np.array([0, 0, 1]),
    ):
        self.translate_frame(translation_vector)
        self.rotate_frame(line_angle, rotation_axis)

    def rotate_frame(self, line_angle: np.ndarray, rotation_axis: np.ndarray):
        self.frame.rotate(line_angle, rotation_axis)
        self.rotate_all(line_angle, rotation_axis)

    def translate_frame(self, translation_vector: np.ndarray):
        self.frame.translate_current(translation_vector)
        self.translate_all(translation_vector)

    def coords_given_frame(self, frame):
        pass
