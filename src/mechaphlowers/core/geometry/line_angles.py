# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.geometry.rotation import (
    rotation_quaternion_same_axis,
)

# class LineAngles:
#     def __init__(self, section_array: SectionArray):
#         self.section_array = section_array


def get_supports_ground_coords(
    span_length: np.ndarray, line_angle: np.ndarray
) -> np.ndarray:
    """Get the coordinates of the supports in the global frame."""
    line_angle_sums = np.cumsum(line_angle)
    coords_supports = np.zeros((span_length.size, 3))
    translations_vectors = np.zeros((span_length.size, 3))
    translations_vectors[:, 0] = span_length
    translations_vectors = rotation_quaternion_same_axis(
        translations_vectors,
        line_angle_sums,
        rotation_axis=np.array([0, 0, 1]),
    )
    coords_supports = np.cumsum(translations_vectors, axis=0)
    coords_supports = np.roll(coords_supports, 1, axis=0)
    coords_supports[0, :] = np.array([0, 0, 0])
    return coords_supports


def build_supports(
    coords_supports: np.ndarray,
    conductor_attachment_altitude: np.ndarray,
    line_angle: np.ndarray,
    crossarm_length: np.ndarray,
) -> np.ndarray:
    """Build the supports in the global frame."""
    center_arm_coords = coords_supports.copy()
    center_arm_coords[:, 2] = conductor_attachment_altitude
    line_angle_sums = np.cumsum(line_angle)

    arm_translation_vectors = np.zeros((line_angle.size, 3))
    arm_translation_vectors[:, 1] = crossarm_length
    arm_translation_vectors = rotation_quaternion_same_axis(
        arm_translation_vectors,
        line_angle_sums,
        rotation_axis=np.array([0, 0, 1]),
    )
    arm_translation_vectors = rotation_quaternion_same_axis(
        arm_translation_vectors,
        -line_angle / 2,
        rotation_axis=np.array([0, 0, 1]),
    )
    return center_arm_coords, center_arm_coords + arm_translation_vectors


def get_supports_layer(
    supports_ground_coords: np.ndarray,
    center_arm_coords: np.ndarray,
    arm_coords: np.ndarray,
) -> np.ndarray:
    """Get the supports in the global frame."""
    support_layers = (
        np.zeros(
            (
                4,
                supports_ground_coords.shape[0],
                supports_ground_coords.shape[1],
            )
        )
        * np.nan
    )
    support_layers[0, :, :] = supports_ground_coords
    support_layers[1, :, :] = center_arm_coords
    support_layers[2, :, :] = arm_coords

    return support_layers


def layer_to_plot(supports_layers):
    """Convert the support coordinates to a format suitable for plotting."""
    return supports_layers.reshape(-1, 3, order='F')

def get_lengths_between_supports(
    supports_ground_coords: np.ndarray,
) -> np.ndarray:
    """Get the lengths between the supports."""
    lengths = np.diff(supports_ground_coords)

    return lengths