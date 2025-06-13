# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple

import numpy as np

from mechaphlowers.core.geometry.line_angles import (
    compute_span_azimuth,
    get_attachment_coords,
    get_edge_arm_coords,
    get_elevation_diff_between_supports,
    get_span_lengths_between_supports,
    get_supports_coords,
    get_supports_ground_coords,
)
from mechaphlowers.core.geometry.rotation import rotation_quaternion_same_axis

""" References for the geometry of the line.

Collections of technical functions to transform coordinates from the different frames of the different objects.
"""


def transform_coordinates(
    x_cable: np.ndarray,
    z_cable: np.ndarray,
    beta: np.ndarray,
    altitude: np.ndarray,
    span_length: np.ndarray,
    crossarm_length: np.ndarray,
    insulator_length: np.ndarray,
) -> np.ndarray:
    """Transform cable coordinates from cable frame to global frame

    Args:
            x_cable: Cable x coordinates
            z_cable: Cable z coordinates
            beta: Load angles in degrees
            altitude: Conductor attachment altitudes
            span_length: Span lengths
            crossarm_length: Crossarm lengths
            insulator_length: Insulator lengths

    Returns:
            np.ndarray: Transformed coordinates array in point format (x,y,z)
    """

    # x_cable, z_cable, beta are 2D arrays in the cable frame (origin at lowest point of the cable)

    x_span, y_span, z_span = cable_to_beta_plane(
        x_cable[:, :-1], z_cable[:, :-1], beta=beta[:-1]
    )

    # x_span, y_span, z_span are 3D arrays in the crossarm frame (origin at lowest point of the cable)

    x_span, y_span, z_span = translate_cable_to_support(
        x_span,
        y_span,
        z_span,
        altitude,
        span_length,
        crossarm_length,
        insulator_length,
    )

    # now origin is at the left attachment point of the cable

    x_span, y_span, z_span = cable_to_crossarm_frame(x_span, y_span, z_span, 0)

    # dont forget to flatten the arrays and stack in a 3xNpoints array
    # Ex: z_span = array([[10., 20., 30.], [11., 12. ,13.]]) -> z_span.reshape(-1) = array([10., 20., 30., 11., 12., 13.])

    return np.vstack(
        [x_span.T.reshape(-1), y_span.T.reshape(-1), z_span.T.reshape(-1)]
    ).T


def cable_to_crossarm_frame(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, alpha: np.ndarray
):
    x0 = x[0, :]
    y0 = y[0, :]

    x = x - x0
    y = y - y0

    vector = spans_to_vector(x, y, z)
    init_shape = z.shape
    span = rotation_quaternion_same_axis(
        vector,
        alpha.repeat(init_shape[0]),  # idea : beta = [b0,..,b0, b1,..,b1,..]
        np.array([0, 0, 1]),
    )  # "z" axis

    x_span, y_span, z_span = (
        span[:, 0].reshape(init_shape, order='F'),
        span[:, 1].reshape(init_shape, order='F'),
        span[:, 2].reshape(init_shape, order='F'),
    )
    return x_span, y_span, z_span


def spans_to_vector(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """spans2vector is a function that allows to stack x, y and z arrays into a single array

    spans are a n x d array where n is the number of points per span and d is the number of spans
    vector are a n x 3 array where n is the number of points per span and 3 is the number of coordinates

    Args:
        x (np.ndarray): n x d array spans x coordinates
        y (np.ndarray): n x d array spans y coordinates
        z (np.ndarray): n x d array spans z coordinates

    Returns:
        np.ndarray: 3 x n array vector coordinates
    """

    cc = np.vstack(
        [
            x.reshape(-1, order='F'),
            y.reshape(-1, order='F'),
            z.reshape(-1, order='F'),
        ]
    ).T
    return cc


def cable_to_beta_plane(
    x: np.ndarray,
    z: np.ndarray,
    beta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cable2span cable to span is a function that allows to rotate from cable 2D plan to span 3D frame with an angle beta


    Args:
        x (np.ndarray): n x d array spans x coordinates
        z (np.ndarray): n x d array spans z coordinates
        beta (np.ndarray): n array angle rotation

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - x_span (np.ndarray): Rotated x coordinates in the span 3D frame.
            - y_span (np.ndarray): Rotated y coordinates in the span 3D frame.
            - z_span (np.ndarray): Rotated z coordinates in the span 3D frame.
    """

    init_shape = z.shape
    # Warning here, x and z are shaped as (n point per span, d span)
    # elevation part has the same shape
    # However rotation is applied on [x,y,z] stacked matrix with x vector of shape (n x d, )
    elevation_part = np.linspace(
        tuple(z[0, :].tolist()),
        tuple(z[-1, :].tolist()),
        x.shape[0],
    )

    vector = spans_to_vector(x, 0 * x, z - elevation_part)
    span = rotation_quaternion_same_axis(
        vector,
        beta.repeat(init_shape[0]),  # idea : beta = [b0,..,b0, b1,..,b1,..]
        np.array([1, 0, 0]),
    )  # "x" axis

    x_span, y_span, z_span = (
        span[:, 0].reshape(init_shape, order='F'),
        span[:, 1].reshape(init_shape, order='F'),
        span[:, 2].reshape(init_shape, order='F'),
    )

    z_span += elevation_part

    return x_span, y_span, z_span


def translate_cable_to_support(
    x_span: np.ndarray,
    y_span: np.ndarray,
    z_span: np.ndarray,
    altitude: np.ndarray,
    span_length: np.ndarray,
    crossarm_length: np.ndarray,
    insulator_length: np.ndarray,
    line_angle: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate cable using altitude and span length

    Args:
        x_span (np.ndarray): x coordinates rotated
        y_span (np.ndarray): y coordinates rotated
        z_span (np.ndarray): z coordinates rotated
        altitude (np.ndarray): conductor heigth altitude
        span_length (np.ndarray): span length
        crossarm_length (np.ndarray): crossarm length
        insulator_length (np.ndarray): insulator length

    Returns:
        Tuple[np.ndarray]: translated x_span, y_span and z_span
    """

    supports_ground_coords = get_supports_ground_coords(
        span_length=span_length,
        line_angle=line_angle,
    )

    _, edge_arm_coords = get_edge_arm_coords(
        supports_ground_coords=supports_ground_coords,
        conductor_attachment_altitude=altitude,
        line_angle=line_angle,
        crossarm_length=crossarm_length,
    )

    attachment_coords = get_attachment_coords(
        edge_arm_coords=edge_arm_coords,
        insulator_length=insulator_length,
    )

    z_span += -z_span[0, :] + attachment_coords[:-1, 2]
    y_span += -y_span[0, :] + attachment_coords[:-1, 1]
    x_span += -x_span[0, :] + attachment_coords[:-1, 0]

    return x_span, y_span, z_span

    # # Note : for every data, we dont need the last support information
    # # Ex : altitude = array([50., 40., 20., 10.]) -> altitude[:-1] = array([50., 40., 20.])
    # # "move" the cable to the conductor attachment altitude
    # z_span += -z_span[0, :] + altitude[:-1]
    # # "move" the cables at the end of the arm
    # y_span += crossarm_length[:-1]
    # # "move down" the cables at the end of the insulator chain
    # z_span += -insulator_length[:-1]
    # # "move" each cable to the x coordinate of the hanging point
    # x_span += -x_span[0, :] + np.pad(
    #     np.cumsum(span_length[:-2]), (1, 0), "constant"
    # )
    # # why pad ? cumsum(...) = array([100., 300.]) and we need a zero to start
    # # pad(...) = array([0., 100., 300.])

    # return x_span, y_span, z_span



class CablePlane:
    """This class handles the parameters for defining the cable plane"""

    def __init__(
        self,
        span_length: np.ndarray,
        conductor_attachment_altitude: np.ndarray,
        crossarm_length: np.ndarray,
        insulator_length: np.ndarray,
        line_angle: np.ndarray,
    ):
        (
            self.supports_ground_coords,
            self.center_arm_coords,
            self.arm_coords,
            self.attachment_coords,
        ) = get_supports_coords(
            span_length,
            line_angle,
            conductor_attachment_altitude,
            crossarm_length,
            insulator_length,
        )

        self.a = span_length
        self.line_angle = line_angle
        self.b = conductor_attachment_altitude
        self.crossarm_length = crossarm_length
        self.insulator_length = insulator_length
        self._beta = np.array([])

    @property
    def a_prime(self) -> np.ndarray:
        return get_span_lengths_between_supports(self.attachment_coords)

    @property
    def b_prime(self) -> np.ndarray:
        return get_elevation_diff_between_supports(self.attachment_coords)

    @property
    def alpha(self) -> np.ndarray:
        return compute_span_azimuth(self.attachment_coords)
