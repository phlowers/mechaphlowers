# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import numpy as np


class Span(ABC):
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
        **kwargs,
    ) -> None:
        self.span_length = span_length
        self.elevation_difference = elevation_difference
        self.sagging_parameter = sagging_parameter
        self.linear_weight = linear_weight
        if load_coefficient is None:
            self.load_coefficient = np.ones_like(span_length)
        else:
            self.load_coefficient = load_coefficient

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
    def x_m(self) -> np.ndarray:
        """Distance between the lowest point of the cable and the left hanging point, projected on the horizontal axis.

        In other words: opposite of the abscissa of the left hanging point.
        """

    @abstractmethod
    def x_n(self) -> np.ndarray:
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
    def L(self) -> np.ndarray:
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
    # Rename this method? This method computes the norm, not necessarily the max. This is only the max for x_m and x_n
    def T_max(self, x_one_per_span: np.ndarray) -> np.ndarray:
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

    # TODO: factorize compute_L and compute_x_n in Span class later ?
    @staticmethod
    @abstractmethod
    def compute_L(
        a: np.ndarray,
        b: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """Computing total length of the cable using a static method"""

    @staticmethod
    @abstractmethod
    def compute_T_h(
        p: np.ndarray, k_load: np.ndarray, lambd: np.float64
    ) -> np.ndarray:
        """Computing horizontal tension on the cable using a static method"""

    @staticmethod
    @abstractmethod
    def compute_p(
        T_h: np.ndarray, k_load: np.ndarray, lambd: np.float64
    ) -> np.ndarray:
        """Computing sagging parameter on the cable using a static method"""

    @staticmethod
    @abstractmethod
    def compute_T_mean(
        a: np.ndarray,
        b: np.ndarray,
        p: np.ndarray,
        T_h: np.ndarray,
    ) -> np.ndarray:
        """Computing mean tension on the cable using a static method"""


class CatenarySpan(Span):
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

    def x_m(self) -> np.ndarray:
        # depedency problem??? use p or T_h?
        a = self.span_length
        b = self.elevation_difference
        p = self.sagging_parameter
        # write if lambd None -> use p instead?
        # return error if linear_weight = None?
        return self.compute_x_m(a, b, p)

    def x_n(self):
        a = self.span_length
        b = self.elevation_difference
        p = self.sagging_parameter
        return self.compute_x_n(a, b, p)

    def x(self, resolution: int = 10) -> np.ndarray:
        """x_coordinate for catenary generation in cable frame

        Args:
        resolution (int, optional): Number of point to generation between supports. Defaults to 10.

        Returns:
        np.ndarray: points generated x number of rows in SectionArray. Last column is nan due to the non-definition of last span.
        """

        start_points = self.x_m()
        end_points = self.x_n()

        return np.linspace(start_points, end_points, resolution)

    def L_m(self) -> np.ndarray:
        a = self.span_length
        b = self.elevation_difference
        p = self.sagging_parameter
        # write if lambd None -> use p instead?
        return self.compute_L_m(a, b, p)

    def L_n(self) -> np.ndarray:
        a = self.span_length
        b = self.elevation_difference
        p = self.sagging_parameter
        return self.compute_L_n(a, b, p)

    def L(self) -> np.ndarray:
        """Total length of the cable."""
        a = self.span_length
        b = self.elevation_difference
        p = self.sagging_parameter
        return self.compute_L(a, b, p)

    def T_h(self) -> np.ndarray:
        if self.linear_weight is None:
            raise AttributeError("Cannot compute T_h: linear_weight is needed")
        else:
            p = self.sagging_parameter
            k_load = self.load_coefficient
            return self.compute_T_h(p, k_load, self.linear_weight)

    def T_v(self, x_one_per_span) -> np.ndarray:
        # an array of abscissa of the same length as the number of spans is expected
        T_h = self.T_h()
        p = self.sagging_parameter
        return self.compute_T_v(x_one_per_span, p, T_h)

    def T_max(self, x_one_per_span) -> np.ndarray:
        # an array of abscissa of the same length as the number of spans is expected
        T_h = self.T_h()
        p = self.sagging_parameter
        return self.compute_T_max(x_one_per_span, p, T_h)

    def T_mean_m(self) -> np.ndarray:
        a = self.span_length
        b = self.elevation_difference
        p = self.sagging_parameter
        T_h = self.T_h()
        return self.compute_T_mean_m(a, b, p, T_h)

    def T_mean_n(self) -> np.ndarray:
        a = self.span_length
        b = self.elevation_difference
        p = self.sagging_parameter
        T_h = self.T_h()
        return self.compute_T_mean_n(a, b, p, T_h)

    def T_mean(self) -> np.ndarray:
        a = self.span_length
        b = self.elevation_difference
        T_h = self.T_h()
        p = self.sagging_parameter
        return self.compute_T_mean(a, b, p, T_h)

    @staticmethod
    def compute_p(
        T_h: np.ndarray, k_load: np.ndarray, lambd: np.float64
    ) -> np.ndarray:
        return T_h / (k_load * lambd)

    @staticmethod
    def compute_x_m(
        a: np.ndarray,
        b: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        return -a / 2 + p * np.arcsinh(b / (2 * p * np.sinh(a / (2 * p))))

    @staticmethod
    def compute_x_n(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
        return a + CatenarySpan.compute_x_m(a, b, p)

    @staticmethod
    def compute_L_m(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> np.ndarray:
        x_m = CatenarySpan.compute_x_m(a, b, p)
        return -p * np.sinh(x_m / p)

    @staticmethod
    def compute_L_n(
        a: np.ndarray,
        b: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        x_n = CatenarySpan.compute_x_n(a, b, p)
        return p * np.sinh(x_n / p)

    # put in superclass?
    @staticmethod
    def compute_L(
        a: np.ndarray,
        b: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        L_m = CatenarySpan.compute_L_m(a, b, p)
        L_n = CatenarySpan.compute_L_n(a, b, p)
        return L_m + L_n

    @staticmethod
    def compute_T_h(
        p: np.ndarray, k_load: np.ndarray, lambd: np.float64
    ) -> np.ndarray:
        return p * k_load * lambd

    @staticmethod
    def compute_T_v(
        x_one_per_span: np.ndarray,
        p: np.ndarray,
        T_h: np.ndarray,
    ) -> np.ndarray:
        # an array of abscissa of the same length as the number of spans is expected
        return T_h * np.sinh(x_one_per_span / p)

    @staticmethod
    def compute_T_max(
        x_one_per_span: np.ndarray,
        p: np.ndarray,
        T_h: np.ndarray,
    ) -> np.ndarray:
        # an array of abscissa of the same length as the number of spans is expected
        return T_h * np.cosh(x_one_per_span / p)

    @staticmethod
    def compute_T_mean_m(
        a: np.ndarray,
        b: np.ndarray,
        p: np.ndarray,
        T_h: np.ndarray,
    ) -> np.ndarray:
        x_m = CatenarySpan.compute_x_m(a, b, p)
        L_m = CatenarySpan.compute_L_m(a, b, p)
        T_max_m = CatenarySpan.compute_T_max(x_m, p, T_h)
        return (-x_m * T_h + L_m * T_max_m) / (2 * L_m)

    @staticmethod
    def compute_T_mean_n(
        a: np.ndarray,
        b: np.ndarray,
        p: np.ndarray,
        T_h: np.ndarray,
    ) -> np.ndarray:
        # Be careful: p and T_h are linked so the input values must be consistent
        x_n = CatenarySpan.compute_x_n(a, b, p)
        L_n = CatenarySpan.compute_L_n(a, b, p)
        T_max_n = CatenarySpan.compute_T_max(x_n, p, T_h)
        return (x_n * T_h + L_n * T_max_n) / (2 * L_n)

    @staticmethod
    def compute_T_mean(
        a: np.ndarray,
        b: np.ndarray,
        p: np.ndarray,
        T_h: np.ndarray,
    ) -> np.ndarray:
        T_mean_m = CatenarySpan.compute_T_mean_m(a, b, p, T_h)
        T_mean_n = CatenarySpan.compute_T_mean_n(a, b, p, T_h)
        L_m = CatenarySpan.compute_L_m(a, b, p)
        L_n = CatenarySpan.compute_L_n(a, b, p)
        L = CatenarySpan.compute_L(a, b, p)
        return (T_mean_m * L_m + T_mean_n * L_n) / L
