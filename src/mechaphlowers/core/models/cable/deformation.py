# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import numpy as np
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.config import options as cfg

IMAGINARY_THRESHOLD = cfg.solver.deformation_imag_thresh  # type: ignore


class IDeformation(ABC):
	"""This abstract class is a base class for models to compute relative cable deformations."""

	def __init__(
		self,
		tension_mean: np.ndarray,
		cable_length: np.ndarray,
		cable_section_area: np.float64,
		linear_weight: np.float64,
		young_modulus: np.float64,
		dilatation_coefficient: np.float64,
		temperature_reference: np.float64,
		polynomial_conductor: Poly,
		sagging_temperature: np.ndarray,
		max_stress: np.ndarray | None = None,
		**kwargs,
	):
		self.tension_mean = tension_mean
		self.cable_length = cable_length
		self.cable_section_area = cable_section_area
		self.linear_weight = linear_weight
		self.young_modulus = young_modulus
		self.dilatation_coefficient = dilatation_coefficient
		self.temp_ref = temperature_reference
		self.polynomial_conductor = polynomial_conductor
		self.sagging_temperature = sagging_temperature

		if max_stress is None:
			self.max_stress = np.full(self.cable_length.shape, 0)

	@abstractmethod
	def L_ref(self) -> np.ndarray:
		"""Unstressed cable length, at a chosen reference temperature"""

	@abstractmethod
	def epsilon(self) -> np.ndarray:
		"""Total relative strain of the cable."""

	@abstractmethod
	def epsilon_mecha(self) -> np.ndarray:
		"""Mechanical part of the relative strain  of the cable."""

	@abstractmethod
	def epsilon_therm(self) -> np.ndarray:
		"""Thermal part of the relative deformation of the cable, compared to a temperature_reference."""

	@staticmethod
	@abstractmethod
	def compute_epsilon_mecha(
		T_mean: np.ndarray,
		E: np.float64,
		S: np.float64,
		polynomial: Poly,
		max_stress: np.ndarray | None = None,
	) -> np.ndarray:
		"""Computing mechanical strain using a static method"""

	@staticmethod
	@abstractmethod
	def compute_epsilon_therm(
		theta: np.ndarray, theta_ref: np.float64, alpha: np.float64
	) -> np.ndarray:
		"""Computing thermal strain using a static method"""


class DeformationRte(IDeformation):
	"""This class implements the deformation model used by RTE."""

	def L_ref(self) -> np.ndarray:
		L = self.cable_length
		epsilon = self.epsilon_therm() + self.epsilon_mecha()
		return L / (1 + epsilon)

	def epsilon_mecha(self) -> np.ndarray:
		T_mean = self.tension_mean
		E = self.young_modulus
		S = self.cable_section_area
		polynomial = self.polynomial_conductor
		return self.compute_epsilon_mecha(
			T_mean, E, S, polynomial, self.max_stress
		)

	def epsilon(self):
		return self.epsilon_mecha() + self.epsilon_therm()

	def epsilon_therm(self) -> np.ndarray:
		sagging_temperature = self.sagging_temperature
		temp_ref = self.temp_ref
		alpha = self.dilatation_coefficient
		return self.compute_epsilon_therm(sagging_temperature, temp_ref, alpha)

	@staticmethod
	def compute_epsilon_mecha(
		T_mean: np.ndarray,
		E: np.float64,
		S: np.float64,
		polynomial: Poly,
		max_stress: np.ndarray | None = None,
	) -> np.ndarray:
		# linear case
		if polynomial.trim().degree() < 2:
			return T_mean / (E * S)
		# polynomial case
		else:
			return DeformationRte.compute_epsilon_mecha_polynomial(
				T_mean, E, S, polynomial, max_stress
			)

	@staticmethod
	def compute_epsilon_mecha_polynomial(
		T_mean: np.ndarray,
		E: np.float64,
		S: np.float64,
		polynomial: Poly,
		max_stress: np.ndarray | None = None,
	) -> np.ndarray:
		"""Computes epsilon when the stress-strain relation is polynomial"""
		sigma = T_mean / S
		if polynomial is None:
			raise ValueError("Polynomial is not defined")
		epsilon_plastic = DeformationRte.compute_epsilon_plastic(
			T_mean, E, S, polynomial, max_stress
		)
		return epsilon_plastic + sigma / E

	@staticmethod
	def compute_epsilon_plastic(
		T_mean: np.ndarray,
		E: np.float64,
		S: np.float64,
		polynomial: Poly,
		max_stress: np.ndarray | None = None,
	) -> np.ndarray:
		"""Computes elastic permanent strain."""
		sigma = T_mean / S
		if max_stress is None:
			max_stress = np.full(T_mean.shape, 0)
		# epsilon plastic is based on the highest value between sigma and max_stress
		highest_constraint = np.fmax(sigma, max_stress)
		equation_solution = DeformationRte.resolve_stress_strain_equation(
			highest_constraint, polynomial
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
		return DeformationRte.find_smallest_real_positive_root(poly_to_resolve)

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

	@staticmethod
	def compute_epsilon_therm(
		theta: np.ndarray, theta_ref: np.float64, alpha: np.float64
	) -> np.ndarray:
		"""Computing thermal strain using a static method"""
		return (theta - theta_ref) * alpha
