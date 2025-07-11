# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Tuple

import numpy as np

from mechaphlowers.core.geometry.line_angles import (
    get_attachment_coords,
    get_edge_arm_coords,
    get_supports_ground_coords,
)
from mechaphlowers.core.geometry.rotation import rotation_quaternion_same_axis

""" References for the geometry of the line.

Collections of technical functions to transform coordinates from the different frames of the different objects.
"""


def cable_to_localsection_frame(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, alpha: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """cable_to_localsection_frame is a function that rotates the cable coordinates from the cable frame to the localsection frame
    The localsection frame is the the section frame with origin at the left support of the cable, but with the same axes than the section frame.

    Args:
        x (np.ndarray): n x d array spans x coordinates
        y (np.ndarray): n x d array spans y coordinates
        z (np.ndarray): n x d array spans z coordinates
        alpha (np.ndarray): absolute angle of the span (degrees)

    Returns:
            x_span: Rotated x coordinates in the localsection frame.
            y_span: Rotated y coordinates in the localsection frame.
            z_span: Rotated z coordinates in the localsection frame.
    """
    x0 = x[0, :]
    y0 = y[0, :]

    x = x - x0
    y = y - y0

    vector = vectors_to_points(x, y, z)
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


def vectors_to_points(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """vectors_to_points is a function that allows to stack x, y and z arrays into a single array

    vectors are a n x d array where n is the number of points per span and d is the number of spans
    points are a n x 3 array where n is the number of points per span and 3 is the number of coordinates

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
    """cable_to_beta_plane is a function that allows to rotate from cable 2D plan to span 3D frame with an angle beta


    Args:
        x (np.ndarray): n x d array spans x coordinates
        z (np.ndarray): n x d array spans z coordinates
        beta (np.ndarray): n array angle rotation

    Returns:
            x_span: Rotated x coordinates in the span 3D frame.
            y_span: Rotated y coordinates in the span 3D frame.
            z_span: Rotated z coordinates in the span 3D frame.
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

    vector = vectors_to_points(x, 0 * x, z - elevation_part)
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
        crossarm_length=crossarm_length,
        line_angle=line_angle,
        insulator_length=insulator_length,
    )

    attachment_coords = get_attachment_coords(edge_arm_coords, altitude)

    z_span += -z_span[0, :] + attachment_coords[:-1, 2]
    y_span += -y_span[0, :] + attachment_coords[:-1, 1]
    x_span += -x_span[0, :] + attachment_coords[:-1, 0]

    return x_span, y_span, z_span

    # # Note : for every data, we dont need the last support information
    # # Ex : altitude = array([50., 40., 20., 10.]) -> altitude[:-1] = array([50., 40., 20.])
