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
    supports_ground_coords = np.zeros((span_length.size, 3))
    translations_vectors = np.zeros((span_length.size, 3))
    translations_vectors[:, 0] = span_length
    translations_vectors = rotation_quaternion_same_axis(
        translations_vectors,
        line_angle_sums,
        rotation_axis=np.array([0, 0, 1]),
    )
    supports_ground_coords = np.cumsum(translations_vectors, axis=0)
    supports_ground_coords = np.roll(supports_ground_coords, 1, axis=0)
    supports_ground_coords[0, :] = np.array([0, 0, 0])
    return supports_ground_coords


def get_edge_arm_coords(
    supports_ground_coords: np.ndarray,
    conductor_attachment_altitude: np.ndarray,
    line_angle: np.ndarray,
    crossarm_length: np.ndarray,
) -> np.ndarray:
    """Build the supports in the global frame."""
    center_arm_coords = supports_ground_coords.copy()
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


def get_attachment_coords(
    edge_arm_coords: np.ndarray,
    insulator_length: np.ndarray,
) -> np.ndarray:
    """Get the coordinates of the attachment points in the global frame."""
    attachment_coords = edge_arm_coords.copy()
    attachment_coords[:, 2] = attachment_coords[:, 2] - insulator_length
    return attachment_coords


def get_supports_layer(
    supports_ground_coords: np.ndarray,
    center_arm_coords: np.ndarray,
    edge_arm_coords: np.ndarray,
    attachment_coords: np.ndarray,
) -> np.ndarray:
    """Get the supports in the global frame."""
    support_layers = (
        np.zeros(
            (
                5,
                supports_ground_coords.shape[0],
                supports_ground_coords.shape[1],
            )
        )
        * np.nan
    )
    support_layers[0, :, :] = supports_ground_coords
    support_layers[1, :, :] = center_arm_coords
    support_layers[2, :, :] = edge_arm_coords
    support_layers[3, :, :] = attachment_coords

    return support_layers


def layer_to_plot(supports_layers):
    """Convert the support coordinates to a format suitable for plotting."""
    return supports_layers.reshape(-1, 3, order='F')


def get_span_lengths_between_supports(
    attachment_coords: np.ndarray,
) -> np.ndarray:
    """Get the lengths between the supports."""
    attachment_coords_x_y = attachment_coords[
        :, :2
    ]  # Keep only x and y coordinates
    # Calculate the lengths between consecutive attachment points
    lengths = np.linalg.norm(
        attachment_coords_x_y - np.roll(attachment_coords_x_y, -1, axis=0),
        axis=1,
    )
    lengths[-1] = np.nan
    return lengths


def get_altitude_diff_between_supports(
    attachment_coords: np.ndarray,
) -> np.ndarray:
    """Get the lengths between the supports."""
    attachment_coords_z = attachment_coords[:, 2]  # Keep only z coordinates
    # Calculate the altitude differences between consecutive attachment points
    alt_diff = abs(
        attachment_coords_z - np.roll(attachment_coords_z, -1, axis=0)
    )
    alt_diff[-1] = np.nan
    return alt_diff


def span_relative_to_absolute_coords(
    attachment_coords: np.ndarray,
    span_coords: np.ndarray,
):
    full_x_axis = np.full_like(attachment_coords, np.array([1, 0, 0]))
    rotation_angles = angle_between_vectors(attachment_coords, full_x_axis)
    span_coords_absolute_frame = rotation_quaternion_same_axis(
        span_coords,
        rotation_angles,
        rotation_axis=np.array([0, 0, 1]),
    )
    # what is the translation vector? Depends on how span_coords are defined
    # need to check transform_coordinates() (references.py)
    translation_vector = attachment_coords
    span_coords_absolute_frame = (
        span_coords_absolute_frame + translation_vector
    )
    return span_coords_absolute_frame


def angle_between_vectors(
    vector_a: np.ndarray, vector_b: np.ndarray
) -> np.ndarray:
    """Calculate the angle between two vectors."""
    dot_product = np.vecdot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a, axis=1)
    norm_b = np.linalg.norm(vector_b, axis=1)
    cos_angle = dot_product / (norm_a * norm_b)
    # Clip the value to avoid numerical errors outside the range [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)
