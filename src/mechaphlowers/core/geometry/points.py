from typing import Self, Tuple  # type: ignore[attr-defined]

import numpy as np

from mechaphlowers.config import options as cfg
from mechaphlowers.core.geometry.line_angles import (
    get_insulator_layer,
    get_supports_layer,
)
from mechaphlowers.core.geometry.references import (
    CablePlane,
    cable_to_beta_plane,
    cable_to_crossarm_frame,
    get_supports_coords,
    translate_cable_to_support,
)
from mechaphlowers.core.models.cable.span import Span


def stack_nan(coords: np.ndarray) -> np.ndarray:
    """Stack NaN values to the coordinates array to ensure consistent shape."""
    stack_array = np.zeros((coords.shape[0], 1, coords.shape[2])) * np.nan
    return np.concatenate((coords, stack_array), axis=1).reshape(
        -1, 3, order='C'
    )


def vectors_to_coords(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.array([x, y, z]).T


def coords_to_points(coords: np.ndarray) -> np.ndarray:
    """Convert the support coordinates to a format suitable for plotting."""
    return coords.reshape(-1, 3, order='F')


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
            return coords_to_points(self.coords)
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

    @classmethod
    def from_vectors(cls, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Self:
        # Mypy does not support the Self type from typing
        """Create Points from a vector of coordinates."""
        if x.ndim != 2 or y.ndim != 2 or z.ndim != 2:
            raise ValueError("x, y, and z must be 2D arrays")

        return cls(vectors_to_coords(x, y, z))

    @classmethod
    def from_coords(cls, coords: np.ndarray) -> Self:
        """Create Points from separate x, y, and z coordinates.
        Args:
            coords (np.ndarray): A 3D array of shape (layers, n_points, 3) where each row is a point (x, y, z).
        """
        return cls(coords)


class SectionPoints:
    def __init__(
        self,
        span_length: np.ndarray,
        conductor_attachment_altitude: np.ndarray,
        crossarm_length: np.ndarray,
        insulator_length: np.ndarray,
        line_angle: np.ndarray,
        span_model: Span,
        **_,
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
        self.init_span(span_model)

    def init_span(self, span_model: Span) -> None:
        self.span_model = span_model
        self.update_ab()
        self.set_cable_coordinates(resolution=cfg.graphics.resolution)

    def update_ab(self):
        self.span_model.span_length = self.plane.a_prime
        self.span_model.elevation_difference = self.plane.b_prime

    def set_cable_coordinates(self, resolution: int) -> None:
        self.x_cable: np.ndarray = self.span_model.x(resolution)
        self.z_cable: np.ndarray = self.span_model.z(self.x_cable)

    @property
    def beta(self) -> np.ndarray:
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

    def span_in_cable_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_span, y_span, z_span = cable_to_beta_plane(
            self.x_cable[:, :-1], self.z_cable[:, :-1], beta=self.beta[:-1]
        )
        return x_span, y_span, z_span

    def span_in_crossarm_frame(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_span, y_span, z_span = self.span_in_cable_frame()
        x_span, y_span, z_span = cable_to_crossarm_frame(
            x_span, y_span, z_span, self.plane.alpha[:-1]
        )
        return x_span, y_span, z_span

    def span_in_section_frame(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
