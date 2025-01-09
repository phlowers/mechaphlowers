# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0from abc import ABC, abstractmethod

from abc import ABC, abstractmethod

import numpy as np


class SpacePositionCableModel(ABC):
	"""This abstract class is a base class for various models describing the cable in its own frame.

	The coordinates are expressed in the cable frame.

	Notes: For now we assume in these space positioning models that there's
	no line angle or wind (or other load on the cable), so we work under the following simplifying assumptions:

	- a = a' = span_length
	- b = b' = elevation_difference

	Support for line angle and wind will be added later.
	"""

	def __init__(
		self,
		span_length: np.ndarray,
		elevation_difference: np.ndarray,
		p: np.ndarray,
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
	def x_m(self) -> np.ndarray:
		"""Distance between the lowest point of the cable and the left hanging point, projected on the horizontal axis.

		In other words: opposite of the abscissa of the left hanging point.
		"""

	def x_n(self) -> np.ndarray:
		"""Distance between the lowest point of the cable and the right hanging point, projected on the horizontal axis.

		In other words: abscissa of the right hanging point.
		"""
		# See above for explanations about following simplifying assumption
		a = self.span_length

		return a + self.x_m()

	@abstractmethod
	def x(self, resolution: int) -> np.ndarray:
		"""x_coordinate for catenary generation in cable frame: abscissa of the different points of the cable

		Args:
		resolution (int, optional): Number of point to generation between supports.

		Returns:
		np.ndarray: points generated x number of rows in SectionArray. Last column is nan due to the non-definition of last span.
		"""


class CatenaryCableModel(SpacePositionCableModel):
	"""Implementation of a space positioning cable model according to the catenary equation.

	The coordinates are expressed in the cable frame.
	"""

	def z(self, x: np.ndarray) -> np.ndarray:
		"""Altitude of cable points depending on the abscissa."""

		# repeating value to perform multidim operation
		xx = x.T
		# self.p is a vector of size (nb support, ). I need to convert it in a matrix (nb support, 1) to perform matrix operation after.
		# Ex: self.p = array([20,20,20,20]) -> self.p([:,new_axis]) = array([[20],[20],[20],[20]])
		pp = self.p[:, np.newaxis]

		rr = pp * (np.cosh(xx / pp) - 1)

		# reshaping back to p,x -> (vertical, horizontal)
		return rr.T

	def x_m(self) -> np.ndarray:
		p = self.p
		# See above for explanations about following simplifying assumptions
		a = self.span_length
		b = self.elevation_difference

		return -a / 2 + p * np.asinh(b / (2 * p * np.sinh(a / (2 * p))))

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
