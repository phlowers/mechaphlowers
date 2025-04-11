# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import numpy as np
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.entities.data_container import DataCable

IMAGINARY_THRESHOLD = 1e-5


class IDeformation(ABC):
	"""This abstract class is a base class for models to compute relative cable deformations."""

	def __init__(
		self,
		tension_mean: np.ndarray,
		cable_length: np.ndarray,
		data_cable: DataCable,
		max_stress: np.ndarray | None = None,
		**kwargs,
	):
		self.tension_mean = tension_mean
		self.cable_length = cable_length
		self.data_cable = data_cable

		if max_stress is None:
			self.max_stress = np.full(self.cable_length.shape, 0)

	@abstractmethod
	def L_ref(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Unstressed cable length, at a chosen reference temperature"""

	@abstractmethod
	def epsilon(
		self,
		current_temperature: np.ndarray,
	) -> np.ndarray:
		"""Total relative strain of the cable."""

	@abstractmethod
	def epsilon_mecha(self) -> np.ndarray:
		"""Mechanical part of the relative strain  of the cable."""

	@abstractmethod
	def epsilon_therm(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Thermal part of the relative deformation of the cable, compared to a temperature_reference."""

	@staticmethod
	@abstractmethod
	def compute_epsilon_mecha(
		T_mean: np.ndarray,
		data_cable: DataCable,
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

	def L_ref(self, current_temperature: np.ndarray) -> np.ndarray:
		L = self.cable_length
		epsilon = (
			self.epsilon_therm(current_temperature) + self.epsilon_mecha()
		)
		return L / (1 + epsilon)

	def epsilon_mecha(self) -> np.ndarray:
		T_mean = self.tension_mean
		E = self.data_cable.young_modulus
		S = self.data_cable.cable_section_area
		return self.compute_epsilon_mecha(
			T_mean, self.data_cable, self.max_stress
		)

	def epsilon(self, current_temperature: np.ndarray):
		return self.epsilon_mecha() + self.epsilon_therm(current_temperature)

	def epsilon_therm(self, current_temperature: np.ndarray) -> np.ndarray:
		temp_ref = self.data_cable.temperature_reference
		alpha = self.data_cable.dilatation_coefficient
		return self.compute_epsilon_therm(current_temperature, temp_ref, alpha)

	@staticmethod
	# epsilon total
	def compute_epsilon_mecha(
		T_mean: np.ndarray,
		data_cable: DataCable,
		max_stress: np.ndarray | None = None,
	) -> np.ndarray:
		# linear case
		if data_cable.polynomial_conductor.trim().degree() < 2:
			E = data_cable.young_modulus
			S = data_cable.cable_section_area
			return T_mean / (E * S)
		# add linear case with two materials

		# polynomial case
		# change way things work here?
		else:
			# test_data
			data_cable.young_modulus_heart
			data_cable.dilatation_coefficient
			data_cable.dilatation_coefficient_conductor
			data_cable.dilatation_coefficient_heart
			data_cable.cable_section_area_conductor
			return DeformationRte.compute_epsilon_mecha_polynomial(
				T_mean, data_cable, max_stress
			)

	@staticmethod
	def compute_epsilon_mecha_polynomial(
		T_mean: np.ndarray,
		data_cable: DataCable,
		max_stress: np.ndarray | None = None,
	) -> np.ndarray:
		"""Computes epsilon when the stress-strain relation is polynomial"""
		S = data_cable.cable_section_area
		E = data_cable.young_modulus
		polynomial_conductor = data_cable.polynomial_conductor
		sigma = T_mean / S
		if polynomial_conductor is None:
			raise ValueError("Polynomial is not defined")
		# if sigma > sigma_max: sum of polynomials
		# else:
		# compute both eps_plastic + eps_th and compare them
		# take the lowest and compute total eps
		# compare total eps from first material to eps_pl + eps_th to other material

		# gérer les histoires de contrainte réduite

		# epsilon_plastic_conductor = DeformationRte.compute_epsilon_plastic(
		# 	T_mean, (E - E_heart), S, polynomial_conductor, max_stress
		# )
		# epsilon_therm_conductor = DeformationRte.compute_epsilon_therm(theta, temp_ref, alpha_conductor)
		# epsilon_plastic_heart = DeformationRte.compute_epsilon_plastic(
		# 	T_mean, E_heart, S, polynomial_heart, max_stress
		# )
		# epsilon_therm_heart = DeformationRte.compute_epsilon_therm(theta, temp_ref, alpha_conductor)
		# epsilon_total_heart = epsilon_plastic_conductor + epsilon_therm_heart + sigma / E
		# # np.where instead of if
		# if epsilon_total_heart < epsilon_plastic_conductor + epsilon_therm_conductor:
		# 	return epsilon_total_heart # or epsilon_mecha
		# else:
		# 	return DeformationRte.compute_epsilon_both_materials()

		epsilon_plastic = DeformationRte.compute_epsilon_plastic(
			T_mean, E, S, polynomial_conductor, max_stress
		)
		return epsilon_plastic + sigma / E

	# @staticmethod
	# def compute_epsilon_both_materials(

	# ) -> np.ndarray:

	# change input here?
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


class SigmaSimple:
	def __init__(self, poly, E, alpha_th, T_labo, epsilon_max=0.0):
		self.poly = poly
		# self.sigma_max = sigma_max
		self.E = E
		self.alpha_th = alpha_th
		self.T_labo = T_labo
		self.epsilon_max = epsilon_max
		self._epsilon_th = 0.0

	@property
	def sigma_max(self):
		return self.sigma_poly(self.epsilon_max)

	@property
	def epsilon_th(self):
		return self._epsilon_th

	@epsilon_th.setter
	def epsilon_th(self, T):
		self._epsilon_th = self.alpha_th * (T - self.T_labo)

	def elastic(self, x):
		return self.E * x + self.sigma_max - self.E * self.epsilon_max

	def sigma(self, x):
		out = self.poly(x - self.epsilon_th)
		out = np.where(
			out < self.sigma_max, self.elastic(x - self.epsilon_th), out
		)
		out = np.where(out < 0, 0, out)
		return out

	def sigma_poly(self, x):
		out = self.poly(x)
		return out
