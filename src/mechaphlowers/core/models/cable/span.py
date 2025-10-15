# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from mechaphlowers.entities.arrays import CableArray, SectionArray


class ISpan(ABC):
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
        **_,
    ) -> None:
        self.span_length = span_length
        self.elevation_difference = elevation_difference
        self.sagging_parameter = sagging_parameter
        self.linear_weight = linear_weight
        if load_coefficient is None:
            self.load_coefficient = np.ones_like(span_length)
        else:
            self.load_coefficient = load_coefficient
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

    def z_many_points(self, x: np.ndarray) -> np.ndarray:
        """Altitude of cable points depending on the abscissa. Many points per spans, used for graphs."""

        # repeating value to perform multidim operation
        xx = x.T
        # self.p is a vector of size (nb support, ). I need to convert it in a matrix (nb support, 1) to perform matrix operation after.
        # Ex: self.p = array([20,20,20,20]) -> self.p([:,new_axis]) = array([[20],[20],[20],[20]])
        pp = self.sagging_parameter[:, np.newaxis]
        # pp = Th / (load_coef * linear_weight) ?

        rr = pp * (np.cosh(xx / pp) - 1)

        # reshaping back to p,x -> (vertical, horizontal)
        return rr.T

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
