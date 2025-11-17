from typing import Callable, Self, Tuple

import numpy as np
from typing_extensions import Literal  # type: ignore[attr-defined]

from mechaphlowers.config import options as cfg
from mechaphlowers.core.geometry.line_angles import (
    CablePlane,
    get_attachment_coords,
    get_insulator_layer,
    get_supports_coords,
    get_supports_layer,
)
from mechaphlowers.core.geometry.references import (
    cable_to_beta_plane,
    cable_to_localsection_frame,
    project_coords,
    translate_cable_to_support_from_attachments,
)
from mechaphlowers.core.models.cable.span import ISpan
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import SectionArray


def stack_nan(coords: np.ndarray) -> np.ndarray:
    """Stack NaN values to the coords array to ensure consistent shape when plot and separate layers in a 2D array."""
    stack_array = np.zeros((coords.shape[0], 1, coords.shape[2])) * np.nan
    return np.concatenate((coords, stack_array), axis=1).reshape(
        -1, 3, order='C'
    )


def vectors_to_coords(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    """Convert 3 vectors of coordinates into an array of points.

    Takes 3 numpy arrays representing x, y, and z coordinates and combines them into a single array of 3D points.
    The input vector format is expected to be (N, L) where N is the number of points per layer and L is the number of layers.
    The output will be an array of shape (number of layers, number of points, 3) where each row represents a point in 3D space.

    Args:
        x (np.ndarray): Array of x-coordinates (N, L)
        y (np.ndarray): Array of y-coordinates (N, L)
        z (np.ndarray): Array of z-coordinates (N, L)

    Returns:
        np.ndarray: Array of points with shape (L,N,3) where N is the length of input vectors

    """
    return np.array([x, y, z]).T


def coords_to_points(coords: np.ndarray) -> np.ndarray:
    """Convert the support coordinates to a format suitable for plotting.

    Args:
        coords (np.ndarray): A 3D array of shape (layers, n_points, 3) where each row is a point (x, y, z).

    Returns:
        np.ndarray: A 2D array of shape (number of points, 3) where each row is a point (x, y, z).
    """
    return coords.reshape(-1, 3, order='F')


class Points:
    """This class handles a set of points in 3D space, represented as a 3D numpy array.
    The points are stored in a 3D array with shape (number of layers, number of points, 3),
    where the last dimension represents the x, y, and z coordinates of each point.

    It provides methods to convert the coordinates to vectors, points, and to create a Points object from.

    Do not use this class directly, use the factory methods `from_vectors` or `from_coords`.

    Examples:
        >>> points = Points.from_vectors(x, y, z)
        >>> span0, span1, span2, ... = points.coords
        >>> span0
        array([[x0, y0, z0],
            [x1, y1, z1],
            ...
            ])
    """

    def __init__(self, coords: np.ndarray):
        if coords.ndim != 3 or coords.shape[2] != 3:
            raise ValueError(
                "Coordinates must be a 3D array with shape (number of layers, number of points, 3)"
            )
        self.coords = coords

    @property
    def vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert the coordinates to vectors. Returns the x, y, z coordinates as separate 2D arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Three 2D arrays representing the x, y, and z coordinates.

        Examples:
            >>> x, y, z = points.vectors
            >>> x
            array([[x0_span0, x0_span1, x0_span2],
                [x1_span0, x1_span1, x1_span2],
                ...
                ])

        """
        return (
            self.coords[:, :, 0].T,
            self.coords[:, :, 1].T,
            self.coords[:, :, 2].T,
        )

    def points(self, stack=False) -> np.ndarray:
        """Convert the coordinates to a 2D array of points for plotting or other uses.

        Args:
            stack (bool, optional): If True, stack NaN values to separate spans when plotting. Defaults to False.

        Returns:
            np.ndarray: A 2D array of shape (number of points, 3) where each row is a point (x, y, z).

        Examples:
            >>> points_array = points.points(stack=False)
            >>> points_array
            array([[x0, y0, z0],
                [x1, y1, z1],
                [x2, y2, z2],
                ...
                ])
            >>> points_array_stacked = points.points(stack=True)
            >>> points_array_stacked
            array([[x0_span0, y0_span0, z0_span0],
                [x1_span0, y1_span0, z1_span0],
                [x2_span0, y2_span0, z2_span0],
                ...
                [nan, nan, nan]
                [x0_span1, y0_span1, z0_span1],
                ...
                ])
        """
        if stack is False:
            return coords_to_points(self.coords)
        else:
            return stack_nan(self.coords)

    def flat_layer(self) -> np.ndarray:
        """Convert the coordinates to a 2D array of points with a column dedicated to layer number for plotting or other uses as dataframe usage.

        Not implemented yet

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
        """Create Points from a vector of coordinates.

        Args:
            x (np.ndarray): Array of x-coordinates (N, L).
            y (np.ndarray): Array of y-coordinates (N, L).
            z (np.ndarray): Array of z-coordinates (N, L).

        Returns:
            Points: An instance of the Points class containing the coordinates.

        Raises:
            ValueError: If x, y, or z are not 2D arrays.
        """
        if x.ndim != 2 or y.ndim != 2 or z.ndim != 2:
            raise ValueError("x, y, and z must be 2D arrays")

        return cls(vectors_to_coords(x, y, z))

    @classmethod
    def from_coords(cls, coords: np.ndarray) -> Self:
        """Create Points from separate x, y, and z coordinates.
        Args:
            coords (np.ndarray): A 3D array of shape (layers, n_points, 3) where each row is a point (x, y, z).

        Returns:
            Points: An instance of the Points class containing the coordinates.
        """
        return cls(coords)


# class SectionPoints:
#     def __init__(
#         self,
#         span_length: np.ndarray,
#         conductor_attachment_altitude: np.ndarray,
#         crossarm_length: np.ndarray,
#         insulator_length: np.ndarray,
#         line_angle: np.ndarray,
#         span_model: ISpan,
#         **_,
#     ):
#         """Initialize the SectionPoints object with section parameters and a span model.

#         Args:
#             span_length (np.ndarray): The length of the spans.
#             conductor_attachment_altitude (np.ndarray): The altitude of the conductor attachments.
#             crossarm_length (np.ndarray): The length of the crossarms.
#             insulator_length (np.ndarray): The length of the insulators.
#             line_angle (np.ndarray): The relative angle of the span.
#             span_model (Span): The span model to use for the points generation.
#         """
#         (
#             self.supports_ground_coords,
#             self.center_arm_coords,
#             self.arm_coords,
#             self.attachment_coords,
#         ) = get_supports_coords(
#             span_length,
#             line_angle,
#             conductor_attachment_altitude,
#             crossarm_length,
#             insulator_length,
#         )
#         self.plane = CablePlane(
#             span_length,
#             conductor_attachment_altitude,
#             crossarm_length,
#             insulator_length,
#             line_angle,
#         )

#         # self.a = span_length
#         self.line_angle = line_angle
#         # self.b = conductor_attachment_altitude
#         self.crossarm_length = crossarm_length
#         self.insulator_length = insulator_length
#         self._beta = np.array([])
#         self.init_span(span_model)

#     def init_span(self, span_model: ISpan) -> None:
#         """change the span model and update the cable coordinates."""
#         self.span_model = span_model
#         self.update_ab()
#         self.set_cable_coordinates(resolution=cfg.graphics.resolution)

#     def update_ab(self):
#         """Sometimes plane object is updated, so we need to update the span model."""
#         self.span_model.span_length = self.plane.a_chain
#         self.span_model.elevation_difference = self.plane.b_chain

#     def set_cable_coordinates(self, resolution: int) -> None:
#         """Set the span in the cable frame 2D coordinates based on the span model and resolution."""
#         self.x_cable: np.ndarray = self.span_model.x(resolution)
#         self.z_cable: np.ndarray = self.span_model.z_many_points(self.x_cable)

#     @property
#     def beta(self) -> np.ndarray:
#         """Get the beta angles for the cable spans.
#         Beta is the angle du to the load on the cable"""
#         if self._beta.size == 0:
#             beta = np.zeros(self.x_cable.shape[1])
#         else:
#             beta = self._beta
#         return beta

#     @beta.setter
#     def beta(self, value: np.ndarray):
#         if not isinstance(value, np.ndarray):
#             raise TypeError("Beta must be a numpy array")
#         if value.ndim != 1:
#             raise ValueError("Beta must be a 1D array")
#         self._beta = value

#     def span_in_cable_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """Get spans as vectors in the cable frame."""
#         # Rotate the cable with an angle to represent the wind
#         x_span, y_span, z_span = cable_to_beta_plane(
#             self.x_cable[:, :-1], self.z_cable[:, :-1], beta=self.beta[:-1]
#         )
#         return x_span, y_span, z_span

#     def span_in_localsection_frame(
#         self,
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """Get spans as vectors in the localsection frame."""
#         x_span, y_span, z_span = self.span_in_cable_frame()
#         x_span, y_span, z_span = cable_to_localsection_frame(
#             x_span, y_span, z_span, self.plane.angle_proj[:-1]
#         )
#         return x_span, y_span, z_span

#     def span_in_section_frame(
#         self,
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """Get spans as vectors in the section frame."""
#         x_span, y_span, z_span = self.span_in_localsection_frame()
#         x_span, y_span, z_span = translate_cable_to_support(
#             x_span,
#             y_span,
#             z_span,
#             self.plane.conductor_attachment_altitude,
#             self.plane.a,
#             self.crossarm_length,
#             self.insulator_length,
#             self.line_angle,
#         )
#         return x_span, y_span, z_span

#     def get_spans(
#         self, frame: Literal["cable", "localsection", "section"]
#     ) -> Points:
#         """get_spans

#         Get the spans Points in the specified frame.

#         Args:
#             frame (Literal['cable', 'localsection', 'section']): frame

#         Raises:
#             ValueError: If the frame is not one of 'cable', 'localsection', or 'section'.

#         Returns:
#             Points: Points object containing the spans in the specified frame.
#         """
#         if frame == "cable":
#             x_span, y_span, z_span = self.span_in_cable_frame()
#         elif frame == "localsection":
#             x_span, y_span, z_span = self.span_in_localsection_frame()
#         elif frame == "section":
#             x_span, y_span, z_span = self.span_in_section_frame()
#         else:
#             raise ValueError(
#                 "Frame must be 'cable', 'localsection' or 'section'"
#             )

#         return Points.from_vectors(x_span, y_span, z_span)

#     def get_supports(self) -> Points:
#         """Get the supports in the section frame."""
#         supports_layers = get_supports_layer(
#             self.supports_ground_coords,
#             self.center_arm_coords,
#             self.arm_coords,
#         )
#         return Points.from_coords(supports_layers)

#     def get_insulators(self) -> Points:
#         """Get the insulators in the section frame."""
#         insulator_layers = get_insulator_layer(
#             self.arm_coords,
#             self.attachment_coords,
#         )
#         return Points.from_coords(insulator_layers)


class SectionPoints:
    def __init__(
        self,
        section_array: SectionArray,
        span_model: ISpan,
        cable_loads: CableLoads,
        get_displacement: Callable,
        **_,
    ):
        """Initialize the SectionPoints object with section parameters and a span model.

        Args:
            section_array (SectionArray): section array
            span_model (ISpan): The span model to use for the points generation.
            cable_loads (CableLoads): cable loads, used for beta angle
            get_displacement (Callable): function that returns an array of chain displacement. Usually, comes from BalanceModel.get_displacement()
        """
        self.cable_loads = cable_loads
        self.section_array = section_array
        span_length = section_array.data.span_length.to_numpy()
        conductor_attachment_altitude = (
            section_array.data.conductor_attachment_altitude.to_numpy()
        )
        crossarm_length = section_array.data.crossarm_length.to_numpy()
        insulator_length = section_array.data.insulator_length.to_numpy()
        line_angle = section_array.data.line_angle.to_numpy()
        ground_altitude = section_array.data.ground_altitude.to_numpy()

        self.plane = CablePlane(
            span_length,
            conductor_attachment_altitude,
            crossarm_length,
            insulator_length,
            line_angle,
            beta=cable_loads.load_angle,
            get_displacement=get_displacement,
            get_attachments_coords=self.get_attachments_coords,
        )

        (
            self.supports_ground_coords,
            self.center_arm_coords,
            self.edge_arm_coords,
            self.attachment_coords,
        ) = get_supports_coords(
            span_length,
            line_angle,
            conductor_attachment_altitude,
            crossarm_length,
            insulator_length,
            self.plane.displacement_vector.dxdydz_global_frame,
            ground_altitude,
        )

        self.line_angle = line_angle
        self.crossarm_length = crossarm_length
        self.insulator_length = insulator_length
        self.init_span(span_model)

    def init_span(self, span_model: ISpan) -> None:
        """change the span model and update the cable coordinates."""
        self.span_model = span_model
        self.set_cable_coordinates(resolution=cfg.graphics.resolution)

    def set_cable_coordinates(self, resolution: int) -> None:
        """Set the span in the cable frame 2D coordinates based on the span model and resolution."""
        self.x_cable: np.ndarray = self.span_model.x(resolution)
        self.z_cable: np.ndarray = self.span_model.z_many_points(self.x_cable)

    def get_attachments_coords(self):
        self.attachment_coords = get_attachment_coords(
            self.edge_arm_coords,
            self.plane.displacement_vector.dxdydz_global_frame,
        )
        return self.attachment_coords

    @property
    def beta(self) -> np.ndarray:
        """Get the beta angles for the cable spans.
        Beta is the angle du to the load on the cable"""
        return self.cable_loads.load_angle

    def span_in_cable_frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get spans as vectors in the cable frame."""
        # Rotate the cable with an angle to represent the wind
        x_span, y_span, z_span = cable_to_beta_plane(
            self.x_cable[:, :-1],
            self.z_cable[:, :-1],
            self.beta[:-1],
            self.plane.a_chain[:-1],
            self.plane.b_chain[:-1],
        )
        return x_span, y_span, z_span

    def span_in_localsection_frame(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get spans as vectors in the localsection frame."""
        x_span, y_span, z_span = self.span_in_cable_frame()
        x_span, y_span, z_span = cable_to_localsection_frame(
            x_span, y_span, z_span, self.plane.azimuth_angle[:-1]
        )
        return x_span, y_span, z_span

    def span_in_section_frame(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get spans as vectors in the section frame."""
        x_span, y_span, z_span = self.span_in_localsection_frame()
        x_span, y_span, z_span = translate_cable_to_support_from_attachments(
            x_span,
            y_span,
            z_span,
            self.get_attachments_coords(),
        )
        return x_span, y_span, z_span

    def get_spans(
        self, frame: Literal["cable", "localsection", "section"]
    ) -> Points:
        """get_spans

        Get the spans Points in the specified frame.

        Args:
            frame (Literal['cable', 'localsection', 'section']): frame

        Raises:
            ValueError: If the frame is not one of 'cable', 'localsection', or 'section'.

        Returns:
            Points: Points object containing the spans in the specified frame.
        """
        self.set_cable_coordinates(resolution=cfg.graphics.resolution)
        if frame == "cable":
            x_span, y_span, z_span = self.span_in_cable_frame()
        elif frame == "localsection":
            x_span, y_span, z_span = self.span_in_localsection_frame()
        elif frame == "section":
            x_span, y_span, z_span = self.span_in_section_frame()
        else:
            raise ValueError(
                "Frame must be 'cable', 'localsection' or 'section'"
            )

        return Points.from_vectors(x_span, y_span, z_span)

    def get_supports(self) -> Points:
        """Get the supports in the section frame."""
        supports_layers = get_supports_layer(
            self.supports_ground_coords,
            self.center_arm_coords,
            self.edge_arm_coords,
        )
        return Points.from_coords(supports_layers)

    def get_insulators(self) -> Points:
        """Get the insulators in the section frame."""
        insulator_layers = get_insulator_layer(
            self.edge_arm_coords,
            self.get_attachments_coords(),
        )
        return Points.from_coords(insulator_layers)

    def get_points_for_plot(
        self, project=False, frame_index=0
    ) -> Tuple[Points, Points, Points]:
        """Get Points objects for span, supports and insulators.
        Can be used for plotting 2D or 3D graphs.

        Args:
            project (bool, optional): Set to True if 2d graph: this project all objects into a support frame. Defaults to False.
            frame_index (int, optional): Index of the frame the projection is made. Should be between 0 and nb_supports-1 included. Unused if project is set to False. Defaults to 0.

        Returns:
            Tuple[Points, Points, Points]: Points for spans, supports and insulators respectively.

        Raises:
            ValueError: frame_index is out of range
        """
        spans_points = self.get_spans("section")
        supports_points = self.get_supports()
        insulators_points = self.get_insulators()
        if project:
            if frame_index > spans_points.coords.shape[0]:
                raise ValueError(
                    f"frame_index out of range. Expected value between 0 and {spans_points.coords.shape[0]}, received {frame_index}"
                )
            spans_points, supports_points, insulators_points = (
                self.project_to_selected_frame(
                    spans_points,
                    supports_points,
                    insulators_points,
                    frame_index,
                )
            )
        return spans_points, supports_points, insulators_points

    def project_to_selected_frame(
        self,
        spans_points: Points,
        supports_points: Points,
        insulators_points: Points,
        frame_index: int,
    ) -> Tuple[Points, Points, Points]:
        """Project spans, supports and insulators points into a support frame.

        Args:
            spans_points (Points): spans Points object
            supports_points (Points): supports Points object
            insulators_points (Points): insulators Points object
            frame_index (int): Index of the frame the projection is made.

        Returns:
            Tuple[Points, Points, Points]: Points for spans, supports and insulators respectively,
            projected into the frame of support number `frame_index`.
        """
        angle_to_project = np.cumsum(self.line_angle)[frame_index]
        translation_vector = -supports_points.coords[frame_index, 0]

        new_span = self.change_frame(
            spans_points, translation_vector, angle_to_project
        )
        new_supports = self.change_frame(
            supports_points, translation_vector, angle_to_project
        )
        new_insulators = self.change_frame(
            insulators_points, translation_vector, angle_to_project
        )

        return new_span, new_supports, new_insulators

    # convert to function? self unused
    def change_frame(
        self,
        points: Points,
        translation_vector: np.ndarray,
        angle_to_project: np.float64,
    ) -> Points:
        """Change the frame of the given Points by applying a translation and a rotation.

        Args:
            points (Points): points to transform
            translation_vector (np.ndarray): translation vector to apply
            angle_to_project (np.float64): angle of the rotation

        Returns:
            Points: new Points object in the new frame
        """
        points.coords = points.coords + translation_vector
        x, y, z = points.vectors
        x, y = project_coords(x, y, angle_to_project)
        # invert y axis to get more natural view
        return Points.from_vectors(x, -y, z)
