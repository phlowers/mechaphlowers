# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
from typing import Self, Tuple, Type

import numpy as np

from mechaphlowers.entities.arrays import CableArray, SectionArray


class ISpan(ABC):
    # TODO: docstring seems old
    """This abstract class is a base class for various models describing the cable in its own frame.

    The coordinates are expressed in the cable frame.

    Notes: For now we assume in these span models that there's
    no line angle or wind (or other load on the cable), so we work under the following simplifying assumptions:

    - a = a' = span_length
    - b = b' = elevation_difference

    Support for line angle and wind will be added later.
    """

    def __init__(
        self,
        span_length: np.ndarray,
        elevation_difference: np.ndarray,
        sagging_parameter: np.ndarray,
        load_coefficient: np.ndarray | None = None,
        linear_weight: np.float64 | None = None,
        span_index: np.ndarray | None = None,
        span_type: np.ndarray | None = None,
        **_,
    ) -> None:
        # TODO: check the vectors have the same size
        # TODO: check that span_type == 0,1,2

        self.span_length = span_length
        self.elevation_difference = elevation_difference
        self.sagging_parameter = sagging_parameter
        self.linear_weight = linear_weight
        if load_coefficient is None:
            self.load_coefficient = np.ones_like(span_length)
        else:
            self.load_coefficient = load_coefficient
        # span_index refers to the index of the span:
        # is [0, 1, 2, ...], if there is no loads
        # is [0, 1, 1, 2 ...], if there is a load in span number 1, and that ISpan represents this span by two values in the arrays
        if span_index is None:
            self.span_index = np.arange(len(span_length))
        else:
            self.span_index = span_index
        # span_type value of 0, 1 or 2 depending on the relationship with the load on the span:
        # - 0 if default span
        # - 1 if semi-span to the left of a load
        # - 2 if semi-span to the right of a load
        if span_type is None:
            self.span_type = np.full_like(span_length, 0)
        else:
            self.span_type = span_type
        self.compute_values()

    def set_lengths(
        self, span_length: np.ndarray, elevation_difference: np.ndarray
    ):
        """Set value of span_length and elevation_difference and compute x_m, x_n and L.

        Args:
            span_length (np.ndarray): new value for span_length parameter.
            elevation_difference (np.ndarray): new value for elevation_difference parameter.
        """
        self.span_length = span_length
        self.elevation_difference = elevation_difference
        self.compute_values()

    def set_parameter(self, sagging_parameter: np.ndarray):
        """Set value of sagging parameter and compute x_m, x_n and L.

        Args:
            sagging_parameter (np.ndarray): new value for sagging parameter.
        """
        self.sagging_parameter = sagging_parameter
        self.compute_values()

    def compute_values(self):
        """Compute and store values for x_m, x_n and L based on current attributes.
        T_mean depends on these values, so this method should be called before calling T_mean(),
        especially if an attribute has been updated.

        The goal of this implementation is to reduce the number of times compute_x_m, compute_x_n and compute_L
        are called during solver iterations.
        """
        self._x_m = self.compute_x_m()
        self._x_n = self.compute_x_n()
        self._L = self.compute_L()

    def mirror(self, span_model: Self) -> None:
        """Copy attributes from an other ISpan object.
        This method is useful for copying values and keeping the reference of the current object.

        Args:
            span_model (Self): other ISpan object to copy attribute from
        """
        self.span_length = span_model.span_length
        self.elevation_difference = span_model.elevation_difference
        self.sagging_parameter = span_model.sagging_parameter
        self.linear_weight = span_model.linear_weight
        self.load_coefficient = span_model.load_coefficient
        self.span_index = span_model.span_index
        self.span_type = span_model.span_type

    @property
    def x_m(self):
        return self._x_m

    @property
    def x_n(self):
        return self._x_n

    @property
    def L(self):
        return self._L

    def update_from_dict(self, data: dict) -> None:
        """Update the span model with new data.

        Args:
                data (dict): Dictionary containing the new data.
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @abstractmethod
    def z_many_points(self, x: np.ndarray) -> np.ndarray:
        """Altitude of cable points depending on the abscissa.

        Args:
        x: abscissa

        Returns:
        altitudes based on the sag tension parameter "p" stored in the model.


        x is an array of any length.

        Example with 3 spans, named a, b, c:

        `span_length = [500, 600, 700]`

        `p = [2_000, 1_500, 1_000]`

        `x = [x0, x1, x2, x3]`

        Then, the output is:
        ```
                      z = [
                          [z0_a, z0_b, z0_c],
                          [z1_a, z1_b, z1_c],
                          [z2_a, z2_b, z2_c],
                          [z3_a, z3_b, z3_c],
                      ]
        ```
        """

    @abstractmethod
    def z_one_point(self, x: np.ndarray) -> np.ndarray:
        """Altitude of cable point depending on the abscissa. One cable point per span
        If there is 2 spans/ 3 supports:

        `span_length = [500, 600, 700]`
        `p = [2_000, 1_500, 1_000]`
        `x = [x0, x1, x2]`

        Then the output is:
        z = [z0, z1, z2]
        """

    @abstractmethod
    def compute_x_m(self) -> np.ndarray:
        """Distance between the lowest point of the cable and the left hanging point, projected on the horizontal axis.

        In other words: opposite of the abscissa of the left hanging point.
        """

    @abstractmethod
    def compute_x_n(self) -> np.ndarray:
        """Distance between the lowest point of the cable and the right hanging point, projected on the horizontal axis.

        In other words: abscissa of the right hanging point.
        """

    @abstractmethod
    def x(self, resolution: int) -> np.ndarray:
        """x_coordinate for catenary generation in cable frame: abscissa of the different points of the cable

        Args:
        resolution (int, optional): Number of point to generation between supports.

        Returns:
        np.ndarray: points generated x number of rows in SectionArray. Last column is nan due to the non-definition of last span.
        """

    @abstractmethod
    def L_m(self) -> np.ndarray:
        """Length of the left portion of the cable.
        The left portion refers to the portion from the left point to lowest point of the cables"""

    @abstractmethod
    def L_n(self) -> np.ndarray:
        """Length of the right portion of the cable.
        The right portion refers to the portion from the right point to lowest point of the cables"""

    @abstractmethod
    def compute_L(self) -> np.ndarray:
        """Total length of the cable."""

    @abstractmethod
    def get_coords(self, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get x and z coordinates for catenary generation in cable frame

        Returns:
            Tuple[np.ndarray, np.ndarray]: x and z coordinates of the cable
        """

    @abstractmethod
    def T_h(self) -> np.ndarray:
        """Horizontal tension on the cable.
        Right now, this tension is constant all along the cable, but that might not be true for elastic catenary model.

        Raises:
                AttributeError: linear_weight is required
        """

    @abstractmethod
    def T_v(self, x_one_per_span: np.ndarray) -> np.ndarray:
        """Vertical tension on the cable, depending on the abscissa.

        Args:
        x_one_per_span: array of abscissa, one abscissa per span: should be at the same length as span_length/elevation_difference/p

        Example with 3 spans, named a, b, c:

        `span_length = [500, 600, 700]`

        `p = [2_000, 1_500, 1_000]`

        Then, x_one_per_span must be of size 3. Each element refers to one span:

        `x_one_per_span = [x_a, x_b, x_c]`

        Then, the output is:
        `T_v = [T_v(x_a), T_v(x_b), T_v(x_c)]`
        """

    @abstractmethod
    def T(self, x_one_per_span: np.ndarray) -> np.ndarray:
        """Norm of the tension on the cable.
        Same as T_v, x_one_per_span must of same length as the number of spans.

        Args:
        x_one_per_span: array of abscissa, one abscissa per span
        """

    @abstractmethod
    def T_mean_m(self) -> np.ndarray:
        """Mean tension of the left portion of the cable."""

    @abstractmethod
    def T_mean_n(self) -> np.ndarray:
        """Mean tension of the right portion of the cable."""

    @abstractmethod
    def T_mean(self) -> np.ndarray:
        """Mean tension along the whole cable."""


class CatenarySpan(ISpan):
    """Implementation of a span cable model according to the catenary equation.

    The coordinates are expressed in the cable frame.
    """

    def z_many_points_local(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Altitude of cable points depending on the abscissa. Many points per spans, used for graphs."""

        # repeating value to perform multidim operation
        xx = x.T
        # self.p is a vector of size (nb support, ). I need to convert it in a matrix (nb support, 1) to perform matrix operation after.
        # Ex: self.p = array([20,20,20,20]) -> self.p([:,new_axis]) = array([[20],[20],[20],[20]])
        pp = p[:, np.newaxis]
        # pp = Th / (load_coef * linear_weight) ?

        rr = pp * (np.cosh(xx / pp) - 1)

        # reshaping back to p,x -> (vertical, horizontal)
        return rr.T

    def z_many_points(self, x: np.ndarray) -> np.ndarray:
        """Altitude of cable points depending on the abscissa. Many points per spans, used for graphs."""
        return self.z_many_points_local(x, self.sagging_parameter)

    def z_one_point(self, x: np.ndarray) -> np.ndarray:
        z = self.sagging_parameter * (np.cosh(x / self.sagging_parameter) - 1)
        return z

    def compute_x_m(self) -> np.ndarray:
        a = self.span_length
        b = self.elevation_difference
        p = self.sagging_parameter
        # return error if linear_weight = None?
        return -a / 2 + p * np.arcsinh(b / (2 * p * np.sinh(a / (2 * p))))

    def compute_x_n(self):
        # move in superclass?
        a = self.span_length
        return a + self.compute_x_m()

    def x(self, resolution: int = 10) -> np.ndarray:
        """x_coordinate for catenary generation in cable frame

        Args:
        resolution (int, optional): Number of point to generation between supports. Defaults to 10.

        Returns:
        np.ndarray: points generated x number of rows in SectionArray. Last column is nan due to the non-definition of last span.
        """

        start_points = self.compute_x_m()
        end_points = self.compute_x_n()

        return np.linspace(start_points, end_points, resolution)

    def _interpolation(
        self, xq: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        interpolate y values at xq points based on x and y arrays.
        This is a non-vectorized solution

        Args:
            xq (np.ndarray): points to interpolate
            x (np.ndarray): known x points
            y (np.ndarray): known y points
        Returns:
            np.ndarray: interpolated y values at xq points
        """
        return np.array(
            [np.interp(xq[i], x[i], y[i]) for i in range(x.shape[0])]
        )

    def get_coords(self, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get x and z coordinates for catenary generation in cable frame. This method handles different span types in case of virtual nodes and produces an output of the same size than the real number of spans.
        Args:
            resolution (int): Number of points to generate between supports.
        Returns:
            Tuple[np.ndarray, np.ndarray]: x and z coordinates of the cable
        """

        start_points = self.compute_x_m()
        end_points = self.compute_x_n()

        if np.all(self.span_type == 0):
            x = np.linspace(start_points, end_points, resolution)
            z = self.z_many_points_local(x, self.sagging_parameter)
            return x, z
        start_points_0 = start_points[self.span_type == 0]
        end_points_0 = end_points[self.span_type == 0]

        start_points_left = start_points[self.span_type == 1]
        start_points_right = start_points[self.span_type == 2]
        end_points_left = end_points[self.span_type == 1]
        end_points_right = end_points[self.span_type == 2]

        x_left = np.linspace(start_points_left, end_points_left, resolution)
        x_right = np.linspace(start_points_right, end_points_right, resolution)
        x_0 = np.linspace(start_points_0, end_points_0, resolution)

        z_left = self.z_many_points_local(
            x_left, self.sagging_parameter[self.span_type == 1]
        )
        z_right = self.z_many_points_local(
            x_right, self.sagging_parameter[self.span_type == 2]
        )
        z_0 = self.z_many_points_local(
            x_0, self.sagging_parameter[self.span_type == 0]
        )

        # join
        x_load = np.concatenate(
            (x_left, x_right + end_points_left - start_points_right), axis=0
        )
        z_load = np.concatenate(
            (z_left, z_right + z_left[-1, :] - z_right[0, :]), axis=0
        )

        # interpolate
        new_x = np.linspace(
            np.min(x_load, axis=0), np.max(x_load, axis=0), resolution
        )
        new_z = self._interpolation(new_x.T, x_load.T, z_load.T).T

        # we have to replace the endpoints left after interpolation
        idx = np.abs(new_x - end_points_left).argmin(axis=0)

        new_x[idx, np.arange(idx.shape[0])] = end_points_left
        new_z[idx, np.arange(idx.shape[0])] = z_left[-1, :]

        mask_output = np.logical_or(self.span_type == 1, self.span_type == 0)
        x = np.full((resolution, self.span_type.shape[0]), np.nan, dtype=float)
        z = np.full((resolution, self.span_type.shape[0]), np.nan, dtype=float)

        x[:, self.span_type == 1] = new_x
        x[:, self.span_type == 0] = x_0

        z[:, self.span_type == 1] = new_z
        z[:, self.span_type == 0] = z_0

        # Store the mapping between interpolated load positions and spans:
        # - idx: indices in the interpolated x-grid (new_x) corresponding to the
        #   left endpoints of loaded spans,
        # - self.span_index[self.span_type == 1]: indices of the spans that
        #   are actually loaded (span_type == 1).
        # This tuple can be used later to retrieve or post-process load values
        # at the correct positions along the interpolated cable.
        self.load_idx = idx, self.span_index[self.span_type == 1]

        return x[:, mask_output], z[:, mask_output]

    def L_m(self) -> np.ndarray:
        p = self.sagging_parameter
        return -p * np.sinh(self.compute_x_m() / p)

    def L_n(self) -> np.ndarray:
        p = self.sagging_parameter
        return p * np.sinh(self.compute_x_n() / p)

    def compute_L(self) -> np.ndarray:
        # move in superclass?
        """Total length of the cable."""
        p = self.sagging_parameter
        return p * (np.sinh(self._x_n / p) - np.sinh(self._x_m / p))

    def T_h(self) -> np.ndarray:
        if self.linear_weight is None:
            raise AttributeError("Cannot compute T_h: linear_weight is needed")
        else:
            p = self.sagging_parameter
            k_load = self.load_coefficient
            lambd = self.linear_weight
            return p * k_load * lambd

    def T_v(self, x_one_per_span) -> np.ndarray:
        # an array of abscissa of the same length as the number of spans is expected
        p = self.sagging_parameter
        return self.T_h() * np.sinh(x_one_per_span / p)

    def T(self, x_one_per_span) -> np.ndarray:
        # an array of abscissa of the same length as the number of spans is expected
        p = self.sagging_parameter
        return self.T_h() * np.cosh(x_one_per_span / p)

    def T_mean_m(self) -> np.ndarray:
        x_m = self.compute_x_m()
        L_m = self.L_m()
        T_h = self.T_h()
        T_x_m = self.T(x_m)
        return (-x_m * T_h + L_m * T_x_m) / (2 * L_m)

    def T_mean_n(self) -> np.ndarray:
        x_n = self.compute_x_n()
        L_n = self.L_n()
        T_h = self.T_h()
        T_x_n = self.T(x_n)
        return (x_n * T_h + L_n * T_x_n) / (2 * L_n)

    def T_mean(self) -> np.ndarray:
        """Return the mean tension along the whole cable. Used in deformation model to compute mechanical deformation.
        Warning: this method uses stored values of x_m, x_n and L.
        If any attribute has been updated, compute_values() should be called before calling this method.

        """
        p = self.sagging_parameter
        k_load = self.load_coefficient
        lambd = self.linear_weight
        a = self._x_n - self._x_m
        return (
            p
            * k_load
            * lambd
            * (
                a
                + (np.sinh(2 * self._x_n / p) - np.sinh(2 * self._x_m / p))
                * p
                / 2
            )
            / self._L
            / 2
        )


def span_model_builder(
    section_array: SectionArray,
    cable_array: CableArray,
    span_model_type: Type[ISpan],
) -> ISpan:
    """Builds a Span object, using data from ScetionArray and CableArray

    Args:
        section_array (SectionArray): input data (span_length, elevation_difference, sagging_parameter)
        cable_array (CableArray): input data from cable (only linar weight used here)
        span_model_type (Type[Span]): choose the type of span model to use

    Returns:
        Span: span model to return
    """
    span_length = section_array.data.span_length.to_numpy()
    elevation_difference = section_array.data.elevation_difference.to_numpy()
    sagging_parameter = section_array.data.sagging_parameter.to_numpy()
    linear_weight = np.float64(cable_array.data.linear_weight.iloc[0])
    return span_model_type(
        span_length,
        elevation_difference,
        sagging_parameter,
        linear_weight=linear_weight,
    )
