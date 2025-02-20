# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import numpy as np

from mechaphlowers.core.models.cable.span import Span

# oyt this in which file?
class CatenarySpan(Span):
	"""Implementation of a span cable model according to the catenary equation.

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
		# depedency problem??? use p or T_h?
		a = self.span_length
		b = self.elevation_difference
		T_h = self.T_h()
		m = self.load_coefficient
		lambd = self.linear_weight
		# write if lambd None -> use p instead?
		# return error if linear_weight = None?
		return CatenarySpan.compute_x_m(a, b, T_h, m, lambd)

	# Currently defined in Span
	# def x_n(self) -> np.ndarray:

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
		T_h = self.T_h()
		m = self.load_coefficient
		lambd = self.linear_weight
		# write if lambd None -> use p instead?
		return CatenarySpan.compute_L_m(a, b, T_h, m, lambd)

	def L_n(self) -> np.ndarray:
		a = self.span_length
		b = self.elevation_difference
		T_h = self.T_h()
		m = self.load_coefficient
		lambd = self.linear_weight
		return CatenarySpan.compute_L_n(a, b, T_h, m, lambd)

	def T_h(self) -> np.ndarray:
		if self.linear_weight is None:
			raise AttributeError("Cannot compute T_h: linear_weight is needed")
		else:
			p = self.p
			m = self.load_coefficient
			return CatenarySpan.compute_T_h(p, m, self.linear_weight)

	def T_v(self, x_one_per_span) -> np.ndarray:
		# an array of abscissa of the same length as the number of spans is expected
		T_h = self.T_h()
		m = self.load_coefficient
		lambd = self.linear_weight
		return CatenarySpan.compute_T_v(x_one_per_span,	T_h, m, lambd) 

	def T_max(self, x_one_per_span) -> np.ndarray:
		# an array of abscissa of the same length as the number of spans is expected
		T_h = self.T_h()
		m = self.load_coefficient
		lambd = self.linear_weight
		return CatenarySpan.compute_T_max(x_one_per_span, T_h, m, lambd) 

	def T_mean_m(self) -> np.ndarray:
		a = self.span_length
		b = self.elevation_difference
		T_h = self.T_h()
		m = self.load_coefficient
		lambd = self.linear_weight
		return CatenarySpan.compute_T_mean_m(a, b, T_h, m, lambd) 

	def T_mean_n(self) -> np.ndarray:
		a = self.span_length
		b = self.elevation_difference
		T_h = self.T_h()
		m = self.load_coefficient
		lambd = self.linear_weight
		return CatenarySpan.compute_T_mean_n(a, b, T_h, m, lambd) 

	def T_mean(self) -> np.ndarray:
		a = self.span_length
		b = self.elevation_difference
		T_h = self.T_h()
		m = self.load_coefficient
		lambd = self.linear_weight
		return CatenarySpan.compute_T_mean(a, b, T_h, m, lambd) 

	# @staticmethod
	# def compute_z(T_h, m ,lambd) -> np.ndarray:
	# 	"""????"""
	# 	return

	@staticmethod
	def compute_p(
		T_h: np.ndarray, m: np.ndarray, lambd: np.ndarray
	) -> np.ndarray:
		return T_h / (m * lambd)

	@staticmethod
	def compute_x_m(
		a: np.ndarray,
		b: np.ndarray,
		T_h: np.ndarray,
		m: np.ndarray,
		lambd: np.ndarray,
	) -> np.ndarray:
		p = CatenarySpan.compute_p(T_h, m, lambd)
		return -a / 2 + p * np.arcsinh(b / (2 * p * np.sinh(a / (2 * p))))

	@staticmethod
	def compute_x_n(
		a: np.ndarray,
		b: np.ndarray,
		T_h: np.ndarray,
		m: np.ndarray,
		lambd: np.ndarray,
	) -> np.ndarray:
		"""Distance between the lowest point of the cable and the right hanging point, projected on the horizontal axis.

		In other words: abscissa of the right hanging point.
		"""
		return a + CatenarySpan.compute_x_m(a, b, T_h, m, lambd)

	@staticmethod
	def compute_L_m(
		a: np.ndarray,
		b: np.ndarray,
		T_h: np.ndarray,
		m: np.ndarray,
		lambd: np.ndarray,
	) -> np.ndarray:
		p = CatenarySpan.compute_p(T_h, m, lambd)
		x_m = CatenarySpan.compute_x_m(a, b, T_h, m, lambd)
		return -p * np.sinh(x_m / p)

	@staticmethod
	def compute_L_n(
		a: np.ndarray,
		b: np.ndarray,
		T_h: np.ndarray,
		m: np.ndarray,
		lambd: np.ndarray,
	) -> np.ndarray:
		p = CatenarySpan.compute_p(T_h, m, lambd)
		x_n = CatenarySpan.compute_x_m(a, b, T_h, m, lambd)
		return p * np.sinh(x_n / p)

	# Put in superclass?
	@staticmethod
	def compute_L(
		a: np.ndarray,
		b: np.ndarray,
		T_h: np.ndarray,
		m: np.ndarray,
		lambd: np.ndarray,
	) -> np.ndarray:
		L_m = CatenarySpan.compute_L_m(a, b, T_h, m, lambd)
		L_n = CatenarySpan.compute_L_n(a, b, T_h, m, lambd)
		return L_m + L_n

	@staticmethod
	def compute_T_h(
		p: np.ndarray, m: np.ndarray, lambd: np.ndarray
	) -> np.ndarray:
		return p * m * lambd

	@staticmethod
	def compute_T_v(
		x_one_per_span: np.ndarray,
		T_h: np.ndarray,
		m: np.ndarray,
		lambd: np.ndarray,
	) -> np.ndarray:
		# an array of abscissa of the same length as the number of spans is expected

		p = CatenarySpan.compute_p(T_h, m, lambd)
		return T_h * np.sinh(x_one_per_span / p)

	@staticmethod
	def compute_T_max(
		x_one_per_span: np.ndarray,
		T_h: np.ndarray,
		m: np.ndarray,
		lambd: np.ndarray,
	) -> np.ndarray:
		# an array of abscissa of the same length as the number of spans is expected

		p = CatenarySpan.compute_p(T_h, m, lambd)
		return T_h * np.sinh(x_one_per_span / p)

	@staticmethod
	def compute_T_mean_m(
		a: np.ndarray,
		b: np.ndarray,
		T_h: np.ndarray,
		m: np.ndarray,
		lambd: np.ndarray,
	) -> np.ndarray:
		x_m = CatenarySpan.compute_x_m(a, b, T_h, m, lambd)
		L_m = CatenarySpan.compute_L_m(a, b, T_h, m, lambd)
		T_max_m = CatenarySpan.compute_T_max(x_m, T_h, m, lambd)
		return (-x_m * T_h + L_m * T_max_m) / (2 * L_m)

	@staticmethod
	def compute_T_mean_n(
		a: np.ndarray,
		b: np.ndarray,
		T_h: np.ndarray,
		m: np.ndarray,
		lambd: np.ndarray,
	) -> np.ndarray:
		x_n = CatenarySpan.compute_x_n(a, b, T_h, m, lambd)
		L_n = CatenarySpan.compute_L_m(a, b, T_h, m, lambd)
		T_max_n = CatenarySpan.compute_T_max(x_n, T_h, m, lambd)
		return (x_n * T_h + L_n * T_max_n) / (2 * L_n)

	@staticmethod
	def compute_T_mean(
		a: np.ndarray,
		b: np.ndarray,
		T_h: np.ndarray,
		m: np.ndarray,
		lambd: np.ndarray,
	) -> np.ndarray:
		T_mean_m = CatenarySpan.compute_T_mean_m(a, b, T_h, m, lambd)
		T_mean_n = CatenarySpan.compute_T_mean_n(a, b, T_h, m, lambd)
		L_m = CatenarySpan.compute_L_m(a, b, T_h, m, lambd)
		L_n = CatenarySpan.compute_L_n(a, b, T_h, m, lambd)
		L = CatenarySpan.compute_L(a, b, T_h, m, lambd)
		return (T_mean_m * L_m + T_mean_n * L_n) / L
