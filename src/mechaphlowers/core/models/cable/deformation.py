# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import numpy as np
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.entities.arrays import CableArray

IMAGINARY_THRESHOLD = 1e-5


class Deformation(ABC):
	"""This abstract class is a base class for models to compute relative cable deformations."""

	def __init__(
		self,
		# cable_array: CableArray,
		span_length: np.ndarray,
		young_modulus: np.ndarray,
		section: np.ndarray,
		temperature_reference: np.ndarray,
		dilatation_coefficient: np.ndarray,
		linear_weight: np.ndarray,
		# tension_mean: np.ndarray,
		a0: np.ndarray,
		a1: np.ndarray,
		a2: np.ndarray,
		a3: np.ndarray,
		a4: np.ndarray,
  		max_stress: np.ndarray | float = 0.0,
		**kwargs
	):
		# self.cable_array = cable_array
		self.young_modulus = young_modulus
		self.section = section
		self.cable_length = span_length
		self.temp_ref = temperature_reference
		self.dilatation_coefficient = dilatation_coefficient
		self.linear_weight = linear_weight
		self.alpha = dilatation_coefficient

		self.a0 = a0
		self.a1 = a1
		self.a2 = a2
		self.a3 = a3
		self.a4 = a4
		
		# self.tension_mean = tension_mean

		if isinstance(max_stress, float):
			self.max_stress = np.full(self.cable_length.shape, max_stress)
		else:
			self.max_stress = max_stress

	def L_ref(self, current_temperature: np.ndarray, tension_mean: np.ndarray) -> np.ndarray:
		"""Unstressed cable length, at a chosen reference temperature"""
		L = self.cable_length
		epsilon = (
			self.epsilon_therm(current_temperature)
			+ self.epsilon_mecha(tension_mean)
		)
		return L / (1 + epsilon)

	# 	return self.compute_L_ref(
	# 			current_temperature, self.cable_length, self.deformation
	# 		)

	# @staticmethod
	# def compute_L_ref(
	# 	current_temperature: np.ndarray,
	# 	cable_length: np.ndarray,

	# ) -> np.ndarray:
	# 	"""Unstressed cable length, at a chosen reference temperature"""
	# 	L = cable_length
	# 	epsilon = (
	# 		self.epsilon_therm(current_temperature)
	# 		+ self.epsilon_mecha()
	# 	)
	# 	return L / (1 + epsilon)

	@property
	def stress_strain_polynomial(self) -> Poly:
		"""Converts coefficients in the dataframe into polynomial"""
		coefs_poly = np.vstack((self.a0.T, self.a1.T, self.a2.T, self.a3.T, self.a4.T)).T[0]
		return Poly(coefs_poly)

	def epsilon_therm(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Thermal part of the relative deformation of the cable, compared to a temperature_reference."""
		# temp_ref = self.cable_array.data["temperature_reference"].to_numpy()
		# alpha = self.cable_array.data["dilatation_coefficient"].to_numpy()
		return self.compute_epsilon_therm(current_temperature, self.temp_ref, self.alpha)

	@abstractmethod
	def epsilon(
		self,
		current_temperature: np.ndarray,
	) -> np.ndarray:
		"""Total relative strain of the cable."""

	@abstractmethod
	def epsilon_mecha(self) -> np.ndarray:
		"""Mechanical part of the relative strain  of the cable."""

	@staticmethod
	@abstractmethod
	def compute_epsilon_mecha(
		T_mean: np.ndarray,
		E: np.ndarray,
		S: np.ndarray,
		polynomial: Poly | None = None,
		max_stress: np.ndarray | None = None,
	) -> np.ndarray:
		"""Computing mechanical strain using a static method"""

	@staticmethod
	def compute_epsilon_therm(
		theta: np.ndarray, theta_ref: np.ndarray, alpha: np.ndarray
	) -> np.ndarray:
		"""Computing thermal strain using a static method"""
		return (theta - theta_ref) * alpha


class LinearDeformation(Deformation):
	"""This model assumes that mechanical strain is linear with tension."""

	def epsilon_mecha(self, T_mean) -> np.ndarray:
		# T_mean = self.tension_mean
		# E = self.cable_array.data["young_modulus"].to_numpy()
		# S = self.cable_array.data["section"].to_numpy()
		return self.compute_epsilon_mecha(T_mean, self.young_modulus, self.section)

	def epsilon(self, current_temperature):
		return self.epsilon_mecha() + self.epsilon_therm(current_temperature)

	@staticmethod
	def compute_epsilon_mecha(
		T_mean: np.ndarray,
		E: np.ndarray,
		S: np.ndarray,
		polynomial: Poly | None = None,
		max_stress: np.ndarray | None = None,
	) -> np.ndarray:
		# polynomial and max_stress are unused here, but are need for polynomial model
		return T_mean / (E * S)


class PolynomialDeformation(Deformation):
	"""This model assumes that mechanical strain and tension follow a polynomial relation."""

	def epsilon_mecha(self, T_mean) -> np.ndarray:
		# T_mean = self.tension_mean
		S = self.section
		E = self.young_modulus
		polynomial = self.stress_strain_polynomial
		return self.compute_epsilon_mecha(
			T_mean, E, S, polynomial, self.max_stress
		)

	@staticmethod
	def compute_epsilon_mecha(
		T_mean: np.ndarray,
		E: np.ndarray,
		S: np.ndarray,
		polynomial: Poly | None = None,
		max_stress: np.ndarray | None = None,
	) -> np.ndarray:
		sigma = T_mean / S
		if polynomial is None:
			raise ValueError("Polynomial is not defined")
		epsilon_plastic = PolynomialDeformation.compute_epsilon_plastic(
			T_mean, E, S, polynomial, max_stress
		)
		return epsilon_plastic + sigma / E

	@staticmethod
	def compute_epsilon_plastic(
		T_mean: np.ndarray,
		E: np.ndarray,
		S: np.ndarray,
		polynomial: Poly,
		max_stress: np.ndarray | None = None,
	) -> np.ndarray:
		"""Computes elastic permanent strain."""
		sigma = T_mean / S
		if max_stress is None:
			max_stress = np.full(T_mean.shape, 0)
		# epsilon plastic is based on the highest value between sigma and max_stress
		highest_constraint = np.fmax(sigma, max_stress)
		equation_solution = (
			PolynomialDeformation.resolve_stress_strain_equation(
				highest_constraint, polynomial
			)
		)
		equation_solution -= highest_constraint / E
		return equation_solution

	@staticmethod
	def resolve_stress_strain_equation(
		sigma: np.ndarray, polynomial: Poly
	) -> np.ndarray:
		"""Solves $\\sigma = Polynomial(\\varepsilon)$"""
		polynom_array = np.full(sigma.shape, polynomial)
		poly_to_resolve = polynom_array - sigma
		return PolynomialDeformation.find_smallest_real_positive_root(
			poly_to_resolve
		)

	@staticmethod
	def find_smallest_real_positive_root(
		poly_to_resolve: np.ndarray,
	) -> np.ndarray:
		"""Find the smallest root that is real and positive for each polynomial

		Args:
			poly_to_resolve (np.ndarray): array of polynomials to solve

		Raises:
			ValueError: if no real positive root has been found for at least one polynomial.

		Returns:
			np.ndarray: array of the roots (one per polynomial)
		"""
		# Can cause performance issues
		all_roots = [poly.roots() for poly in poly_to_resolve]

		all_roots_stacked = np.stack(all_roots)
		keep_solution_condition = np.logical_and(
			abs(all_roots_stacked.imag) < IMAGINARY_THRESHOLD,
			0.0 <= all_roots_stacked,
		)
		# Replace roots that are not real nor positive by np.inf
		real_positive_roots = np.where(
			keep_solution_condition, all_roots_stacked, np.inf
		)
		real_smallest_root = real_positive_roots.min(axis=1).real
		if np.inf in real_smallest_root:
			raise ValueError("No solution found for at least one span")
		return real_smallest_root

	# When implementing 2 meterials: decide if this should be in superclass instead
	def epsilon(
		self,
		current_temperature: np.ndarray,
	):
		return self.epsilon_mecha() + self.epsilon_therm(current_temperature)
