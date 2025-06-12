# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Self, Tuple

import numpy as np

from mechaphlowers.core.geometry.line_angles import (
    compute_span_azimuth,
    get_attachment_coords,
    get_edge_arm_coords,
    get_elevation_diff_between_supports,
    get_insulator_layer,
    get_span_lengths_between_supports,
    get_supports_coords,
    get_supports_ground_coords,
    get_supports_layer,
    layer_to_plot,
)
from mechaphlowers.core.geometry.rotation import rotation_quaternion_same_axis
from mechaphlowers.core.models.cable.span import Span


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


def vectors_to_coords(x, y, z):
    return np.array([x, y, z]).T


def stack_nan(coords: np.ndarray) -> np.ndarray:
    """Stack NaN values to the coordinates array to ensure consistent shape."""
    stack_array = np.zeros((coords.shape[0], 1, coords.shape[2])) * np.nan
    return np.concatenate((coords, stack_array), axis=1).reshape(
        -1, 3, order='C'
    )


class Points:
    """This class handles a set of points in 3D space, represented as a 3D numpy array.
    The points are stored in a 3D array with shape (number of layers, number of points, 3),
    where the last dimension represents the x, y, and z coordinates of each point.

    It provides methods to convert the coordinates to vectors, points, and to create a Points object from.
    """

    def __init__(self, coords: np.ndarray):
        if coords.ndim != 3 or coords.shape[2] != 3:
            raise ValueError(
                "Coordinates must be a 3D array with shape (number of layers, number of points, 3)"
            )
        self.coords = coords

    @property
    def vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert the coordinates to vectors."""
        return (
            self.coords[:, :, 0].T,
            self.coords[:, :, 1].T,
            self.coords[:, :, 2].T,
        )

    def points(self, stack=False) -> np.ndarray:
        """Convert the coordinates to a 2D array of points for plotting or other uses.
        Returns:
            np.ndarray: A 2D array of shape (number of points, 3) where each row is a point (x, y, z).
        """
        if stack is False:
            return layer_to_plot(self.coords)
        else:
            return stack_nan(self.coords)

    def flat_layer(self) -> np.ndarray:
        """Convert the coordinates to a 2D array of points with a column dedicated to layer number for plotting or other uses as dataframe usage.
        Returns:
            np.ndarray: A 2D array of shape (number of layers x number of points, 4) where each row is a point (num layer, x, y, z).
        """
        raise NotImplementedError

    def __repr__(self):
        return f"Points(coords={self.coords})"

    def __len__(self):
        """Return the number of points."""
        return self.coords.shape[0]

    @staticmethod
    def from_vectors(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Self:
        """Create Points from a vector of coordinates."""
        if x.ndim != 2 or y.ndim != 2 or z.ndim != 2:
            raise ValueError("x, y, and z must be 2D arrays")

        return Points(vectors_to_coords(x, y, z))

    @staticmethod
    def from_coords(coords: np.ndarray) -> Self:
        """Create Points from separate x, y, and z coordinates.
        Args:
            coords (np.ndarray): A 3D array of shape (layers, n_points, 3) where each row is a point (x, y, z).
        """
        return Points(coords)


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
            insulator_length
        )

        self.a = span_length
        self.line_angle = line_angle
        self.b = conductor_attachment_altitude
        self.crossarm_length = crossarm_length
        self.insulator_length = insulator_length
        self._beta = np.array([])

    @property
    def a_prime(self):
        return get_span_lengths_between_supports(self.attachment_coords)

    @property
    def b_prime(self):
        return get_elevation_diff_between_supports(self.attachment_coords)

    @property
    def alpha(self) -> np.ndarray:
        return compute_span_azimuth(self.attachment_coords)


class SectionPoints:
    def __init__(
        self,
        span_length,
        conductor_attachment_altitude,
        crossarm_length,
        insulator_length,
        line_angle, *args, **kwargs
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
        self.plane = CablePlane(
            span_length,
            conductor_attachment_altitude,
            crossarm_length,
            insulator_length,
            line_angle,
        )
        
        # self.a = span_length
        self.line_angle = line_angle
        # self.b = conductor_attachment_altitude
        self.crossarm_length = crossarm_length
        self.insulator_length = insulator_length
        self._beta = np.array([])

    def init_span(self, span_model: Span):
        self.span_model = span_model
        self.update_ab()
        self.set_cable_coordinates()

    def update_ab(self):
        self.span_model.span_length = self.plane.a_prime
        self.span_model.elevation_difference = self.plane.b_prime


    def set_cable_coordinates(self, resolution: int = 7):
        self.x_cable: np.ndarray = self.span_model.x(resolution)
        self.z_cable: np.ndarray = self.span_model.z(self.x_cable)

    @property
    def beta(self):
        if self._beta.size == 0:
            beta = np.zeros(self.x_cable.shape[1])
        else:
            beta = self._beta
        return beta

    @beta.setter
    def beta(self, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            raise TypeError("Beta must be a numpy array")
        if value.ndim != 1:
            raise ValueError("Beta must be a 1D array")
        self._beta = value

    def span_in_cable_frame(self):
        x_span, y_span, z_span = cable_to_beta_plane(
            self.x_cable[:, :-1], self.z_cable[:, :-1], beta=self.beta[:-1]
        )
        return x_span, y_span, z_span

    def span_in_crossarm_frame(self):
        x_span, y_span, z_span = self.span_in_cable_frame()
        x_span, y_span, z_span = cable_to_crossarm_frame(
            x_span, y_span, z_span, self.plane.alpha[:-1]
        )
        return x_span, y_span, z_span

    def span_in_section_frame(self):
        x_span, y_span, z_span = self.span_in_crossarm_frame()
        x_span, y_span, z_span = translate_cable_to_support(
            x_span,
            y_span,
            z_span,
            self.plane.b,
            self.plane.a,
            self.crossarm_length,
            self.insulator_length,
            self.line_angle,
        )
        return x_span, y_span, z_span

    def get_spans(self, frame) -> Points:
        if frame == "cable":
            x_span, y_span, z_span = self.span_in_cable_frame()
        elif frame == "crossarm":
            x_span, y_span, z_span = self.span_in_crossarm_frame()
        elif frame == "section":
            x_span, y_span, z_span = self.span_in_section_frame()
        else:
            raise ValueError("Frame must be 'cable', 'crossarm' or 'section'")

        return Points.from_vectors(x_span, y_span, z_span)

    def get_supports(self) -> Points:
        """Get the supports in the global frame."""
        supports_layers = get_supports_layer(
            self.supports_ground_coords,
            self.center_arm_coords,
            self.arm_coords,
        )
        return Points.from_coords(supports_layers)
    
    def get_insulators(self) -> Points:
        """Get the insulators in the global frame."""
        insulator_layers = get_insulator_layer(
            self.arm_coords,
            self.attachment_coords,
        )
        return Points.from_coords(insulator_layers)
