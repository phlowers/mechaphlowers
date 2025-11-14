# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Callable, Tuple

import numpy as np

from mechaphlowers.core.geometry.rotation import (
    rotation_quaternion_same_axis,
)

"""Line angles module

Collections of technical functions and helpers to take into account angles in the coordinates computation of objects.
"""


def compute_span_azimuth(
    attachment_coords: np.ndarray,
) -> np.ndarray:
    """compute_span_azimuth

    Compute the azimuth angle of the span between two attachment points.
    The azimuth angle is the angle between the x-axis and the line connecting two attachment points in the xy-plane.
    The angle is computed in radians and rotation is counter-clockwise (trigonometric).

    Args:
        attachment_coords (np.ndarray): Attachment coordinates of the span.

    Returns:
        1D array of shape (n,) representing the azimuth angle of the span in radians.
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
    rotation_angles[-1] = np.nan
    return rotation_angles


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
    dot_product = np.vecdot(vector_a, vector_b)
    angle_result = np.arctan2(cross_product, dot_product)
    # Return NaN if either vector is null
    is_vector_null = np.logical_or(
        (vector_a == 0).all(axis=1), (vector_b == 0).all(axis=1)
    )
    return np.where(is_vector_null, np.nan, angle_result)


def get_supports_ground_coords(
    span_length: np.ndarray,
    line_angle: np.ndarray,
    ground_altitude: np.ndarray,
) -> np.ndarray:
    """Get the coordinates of the supports in the global frame. These are the coordinates of the barycenter of the support, at ground.

    Args:
        span_length (np.ndarray): span lengths between supports (input from SectionArray)
        line_angle (np.ndarray): line angles (input from SectionArray)

    Returns:
        2D array of shape (n, 3) representing the coordinates of the supports in the global frame.
    """
    line_angle_sums = np.cumsum(line_angle)
    # Creates the translations vectors: these are the vectors between two supports
    translations_vectors = np.empty((span_length.size, 3))
    translations_vectors[:, 0] = span_length
    translations_vectors[:, 1:2] = 0
    translations_vectors = rotation_quaternion_same_axis(
        translations_vectors,
        line_angle_sums,
        rotation_axis=np.array([0, 0, 1]),
    )
    # Computes the coordinates of the supports by adding successive translation vectors
    supports_ground_coords = np.cumsum(translations_vectors, axis=0)
    supports_ground_coords = np.roll(supports_ground_coords, 1, axis=0)
    # Ensure that the first coordinates are (0,0,0), and not (nan,nan,nan)
    supports_ground_coords[0, :] = np.array([0, 0, 0])
    # Add ground altitudes
    supports_ground_coords[:, 2] = ground_altitude
    return supports_ground_coords


def get_edge_arm_coords(
    supports_ground_coords: np.ndarray,
    conductor_attachment_altitude: np.ndarray,
    crossarm_length: np.ndarray,
    line_angle: np.ndarray,
    insulator_length: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the supports and arms in the global frame.

    Args:
        supports_ground_coords (np.ndarray): coordinates of ground supports (output of `get_supports_ground_coords()`)
        conductor_attachment_altitude (np.ndarray): attachment altitude (input from SectionArray)
        crossarm_length (np.ndarray): crossarm lengths (input from SectionArray)
        line_angle (np.ndarray): line angles (input from SectionArray)
        insulator_length (np.ndarray): insulator lengths (input from SectionArray)


    Returns:
        Returns two 2D arrays of shape (n, 3):
            - center_arm_coords: coordinates of the intersection of arms and supports in the global frame
            - edge_arm_coords: coordinates of the edge of the arms in the global frame
    """
    # Create the coordinates of the intersection of the arms and the supports by adding attachmeent altitude
    center_arm_coords = supports_ground_coords.copy()

    # TODO: to refactor later
    center_arm_coords[:, 2] = conductor_attachment_altitude
    center_arm_coords[1:-1, 2] = (
        conductor_attachment_altitude[1:-1] + insulator_length[1:-1]
    )

    line_angle_sums = np.cumsum(line_angle)
    # Create translation vectors, which are the vectors that follows the arm
    arm_translation_vectors = np.zeros((line_angle.size, 3))
    arm_translation_vectors[:, 1] = crossarm_length
    # Rotate the translation vectors into the global frame
    arm_translation_vectors = rotation_quaternion_same_axis(
        arm_translation_vectors,
        line_angle_sums,
        rotation_axis=np.array([0, 0, 1]),
    )
    # Rotate the translation vectors to take into account the angle of the line
    arm_translation_vectors = rotation_quaternion_same_axis(
        arm_translation_vectors,
        -line_angle / 2,
        rotation_axis=np.array([0, 0, 1]),
    )
    return center_arm_coords, center_arm_coords + arm_translation_vectors


def get_attachment_coords(
    edge_arm_coords: np.ndarray,
    displacement_vector: np.ndarray,
) -> np.ndarray:
    """Get the coordinates of the attachment points in the global frame. These are the coordinates of the end of the suspension insulators.
    Currently, we assume that isulators set are vetical.

    Args:
        edge_arm_coords (np.ndarray): coordinates of the edge of the arms (output of `get_edge_arm_coords()`)
        displacement_vector (np.ndarray): displacement vector of the chains (output of BalanceEngine.change_state())

    Returns:
        np.ndarray: coordinates of the attachment points in the global frame.
    """
    return edge_arm_coords + displacement_vector


def get_supports_layer(
    supports_ground_coords: np.ndarray,
    center_arm_coords: np.ndarray,
    edge_arm_coords: np.ndarray,
) -> np.ndarray:
    """Stack the coordinates of the supports and the arms in the global frame."""

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
    """Stack the coordinates of the insulators in the global frame."""

    return np.stack(
        (
            edge_arm_coords,
            attachment_coords,
        ),
        axis=1,
    )


def get_span_lengths_between_attachments(
    attachment_coords: np.ndarray,
) -> np.ndarray:
    """Get the lengths between the attachment points."""
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


def get_elevation_diff_between_attachments(
    attachment_coords: np.ndarray,
) -> np.ndarray:
    """Get the elevation differences between attachment points."""
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
    displacement_vector: np.ndarray,
    ground_altitude: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Helper to get all the coordinates of the supports packed in a tuple."""
    supports_ground_coords = get_supports_ground_coords(
        span_length, line_angle, ground_altitude
    )
    center_arm_coords, arm_coords = get_edge_arm_coords(
        supports_ground_coords,
        conductor_attachment_altitude,
        crossarm_length,
        line_angle,
        insulator_length,
    )
    attachment_coords = get_attachment_coords(arm_coords, displacement_vector)
    return (
        supports_ground_coords,
        center_arm_coords,
        arm_coords,
        attachment_coords,
    )


class DisplacementVector:
    """Class to store chain displacement, and change frame of displacement vector into the global frame"""

    def __init__(
        self, get_displacement: Callable, line_angle: np.ndarray
    ) -> None:
        self.get_displacement = get_displacement
        self.line_angle = line_angle
        self.change_frame()

    def change_frame(self) -> None:
        """Change frame of displacement vector from support frame to global frame"""
        line_angle_sums = np.cumsum(self.line_angle)

        temp_value = rotation_quaternion_same_axis(
            self.get_displacement(),
            line_angle_sums,
            rotation_axis=np.array([0, 0, 1]),
        )
        # Rotate the translation vectors to take into account the angle of the line
        self._dxdydz_global_frame = rotation_quaternion_same_axis(
            temp_value,
            -self.line_angle / 2,
            rotation_axis=np.array([0, 0, 1]),
        )

    @property
    def dxdydz_global_frame(self) -> np.ndarray:
        """Returns displacement vector dxdydz in the global frame.

        Format: `[[dx0, dy0, dz0], [dx1, dy1, dz1], ...]`

        Returns:
            np.ndarray: displacement vector in the global frame
        """
        self.change_frame()
        return self._dxdydz_global_frame


class CablePlane:
    """This class handles the parameters for defining the cable plane"""

    def __init__(
        self,
        span_length: np.ndarray,
        conductor_attachment_altitude: np.ndarray,
        crossarm_length: np.ndarray,
        insulator_length: np.ndarray,
        line_angle: np.ndarray,
        beta: np.ndarray,
        get_displacement: Callable,
        get_attachments_coords: Callable,
    ):
        self.get_attachments_coords = get_attachments_coords
        self.displacement_vector = DisplacementVector(
            get_displacement, line_angle
        )

        self.a = span_length
        self.line_angle = line_angle
        self.conductor_attachment_altitude = conductor_attachment_altitude
        self.crossarm_length = crossarm_length
        self.insulator_length = insulator_length
        self.beta = beta

    @property
    def attachment_coords(self) -> np.ndarray:
        return self.get_attachments_coords()

    @property
    def b(self):
        """no need here to compute b but for memory // for coherence of the class it could be added one day"""
        raise NotImplementedError

    @property
    def a_chain(self) -> np.ndarray:
        """Span length, taking into account arm, line angles and chain"""
        return get_span_lengths_between_attachments(self.attachment_coords)

    @property
    def b_chain(self) -> np.ndarray:
        """Elevation difference, taking into account arm, line angles and chain"""
        return get_elevation_diff_between_attachments(self.attachment_coords)

    @property
    def a_prime(self) -> np.ndarray:
        """Span length after taking wind angle into account"""
        return (
            self.a_chain**2 + self.b_chain**2 * np.sin(self.beta) ** 2
        ) ** 0.5

    @property
    def b_prime(self) -> np.ndarray:
        """Elevation difference after taking wind angle into account"""
        return self.b_chain * np.cos(self.beta)

    @property
    def azimuth_angle(self) -> np.ndarray:
        """Azimuth angle: horizontal angle between
        the current span (chain and arm included)
        and the first line (the line between the first two supports)"""
        return compute_span_azimuth(self.attachment_coords)

    @property
    def alpha(self) -> np.ndarray:
        return np.arctan((self.b_chain * np.sin(self.beta)) / self.a_chain)
