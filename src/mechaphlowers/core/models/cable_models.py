# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0from abc import ABC, abstractmethod

from abc import ABC, abstractmethod

import numpy as np


class GeometricCableModel(ABC):
    """This abstract class is a base class for various models describing the cable in its own frame.

    The coordinates are expressed in the cable frame.

    Notes: For now we assume in these geometric models that there's
    no line angle or wind (or other load on the cable), so we work under the following simplifying assumptions:

    - a = a' = span_length
    - b = b' = elevation_difference

    Support for line angle and wind will be added later.
    """

    def __init__(
        self, span_length: np.ndarray, elevation_difference: np.ndarray, p: np.ndarray
    ) -> None:
        self.span_length = span_length
        self.elevation_difference = elevation_difference
        self.p = p

    @abstractmethod
    def z(self, x: np.ndarray) -> np.ndarray:
        """Altitude of cable points depending on the abscissa.

        Args:
            x: abscissa

        Returns:
            altitudes based on the sag tension parameter "p" stored in the model.
        """

    @abstractmethod
    def x_m(self) -> float:
        """Distance between the lowest point of the cable and the left hanging point, projected on the horizontal axis.

        In other words: opposite of the abscissa of the left hanging point.
        """

    def x_n(self) -> float:
        """Distance between the lowest point of the cable and the right hanging point, projected on the horizontal axis.

        In other words: abscissa of the right hanging point.
        """
        # See above for explanations about following simplifying assumption
        a_prime = self.elevation_difference

        return a_prime + self.x_m()


class CatenaryCableModel(GeometricCableModel):
    """Implementation of a geometric cable model according to the catenary equation.

    The coordinates are expressed in the cable frame.
    """

    def z(self, x: np.ndarray) -> np.ndarray:
        """Altitude of cable points depending on the abscissa."""
        return self.p * (np.cosh(x / self.p) - 1)

    def x_m(self) -> float:
        p = self.p
        # See above for explanations about following simplifying assumptions
        a = self.span_length
        b = self.elevation_difference

        return -a / 2 + p * np.asinh(b / (2 * p * np.sinh(a / (2 * p))))
