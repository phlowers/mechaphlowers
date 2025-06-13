# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple

import numpy as np

from mechaphlowers.core.geometry.rotation import (
    rotation_quaternion_same_axis,
)

"""Line angles module

Collections of technical functions and helpers to take into account angles in the coordinates computation of objects.
"""

def compute_span_azimuth(
    attachment_coords: np.ndarray,
):
    """compute_span_azimuth 

    Compute the azimuth angle of the span between two attachment points.
    The azimuth angle is the angle between the x-axis and the line connecting two attachment points in the xy-plane.
    The angle is computed in degrees and rotation is counter-clockwise (trigonometric).

    Args:
        attachment_coords (np.ndarray): Attachment coordinates of the span.

    Returns:
        np.ndarray: 1D array of shape (n,) representing the azimuth angle of the span in degrees.
    """
    vector_attachment_to_next = (
        np.roll(attachment_coords[:, :-1], -1, axis=0)
        - attachment_coords[:, :-1]
    )
    full_x_axis = np.full_like(vector_attachment_to_next, np.array([1, 0]))
    rotation_angles = angle_between_vectors(
        full_x_axis,
        vector_attachment_to_next,
    )

    return rotation_angles * 180 / np.pi


def angle_between_vectors(
    vector_a: np.ndarray, vector_b: np.ndarray
) -> np.ndarray:
    """Calculate the angle between two 2D vectors.
    
    Arguments:
        vector_a: A 2D array of shape (n, 2) representing the first set of vectors.
        vector_b: A 2D array of shape (n, 2) representing the second set of vectors.
        
    Returns:
        A 1D array of angles in radians, where each angle corresponds to the angle between the vectors at the same index.
        """
    cross_product = np.cross(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a, axis=1)
    norm_b = np.linalg.norm(vector_b, axis=1)
    sin_angle = cross_product / (norm_a * norm_b)
    # Clip the value to avoid numerical errors outside the range [-1, 1]
    sin_angle = np.clip(sin_angle, -1.0, 1.0)
    return np.arcsin(sin_angle)


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
) -> np.ndarray:
    """Get the supports in the global frame."""

    return np.stack(
        (
            supports_ground_coords,
            center_arm_coords,
            edge_arm_coords,

        ),
        axis=1,
    )
    
def get_insulator_layer(
    edge_arm_coords: np.ndarray,
    attachment_coords: np.ndarray,
) -> np.ndarray:
    """Get the supports in the global frame."""

    return np.stack(
        (
            edge_arm_coords,
            attachment_coords,
        ),
        axis=1,
    )







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


def get_elevation_diff_between_supports(
    attachment_coords: np.ndarray,
) -> np.ndarray:
    """Get the elevation differences between the supports."""
    attachment_coords_z = attachment_coords[:, 2]  # Keep only z coordinates
    # Calculate the altitude differences between consecutive attachment points
    # warning: this is right minus left (z_N - z_M in the span notation)
    alt_diff = np.roll(attachment_coords_z, -1, axis=0) - attachment_coords_z

    alt_diff[-1] = np.nan
    return alt_diff


def get_supports_coords(
    span_length: np.ndarray,
    line_angle: np.ndarray,
    conductor_attachment_altitude: np.ndarray,
    crossarm_length: np.ndarray,
    insulator_length: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper to get all the coordinates of the supports packed in a tuple."""
    supports_ground_coords = get_supports_ground_coords(
        span_length, line_angle
    )
    center_arm_coords, arm_coords = get_edge_arm_coords(
        supports_ground_coords,
        conductor_attachment_altitude,
        line_angle,
        crossarm_length,
    )
    attachment_coords = get_attachment_coords(
        edge_arm_coords=arm_coords, insulator_length=insulator_length
    )
    return (
        supports_ground_coords,
        center_arm_coords,
        arm_coords,
        attachment_coords,
    )


