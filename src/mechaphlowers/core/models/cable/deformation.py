# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import numpy as np

from mechaphlowers.entities.arrays import CableArray

IMAGINARY_THRESHOLD = 1e-5


class Deformation(ABC):
	"""This abstract class is a base class for models to compute relative cable deformations."""

	def __init__(
		self,
		cable_array: CableArray,
		tension_mean: np.ndarray,
	):
		self.cable_array = cable_array
		self.tension_mean = tension_mean

	def epsilon_therm(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Thermal part of the relative deformation of the cable, compared to a temperature_reference."""
		temp_ref = self.cable_array.data["temperature_reference"].to_numpy()
		alpha = self.cable_array.data["dilatation_coefficient"].to_numpy()
		return (current_temperature - temp_ref) * alpha

	@abstractmethod
	def epsilon(
		self,
		current_temperature: np.ndarray,
		max_stress: np.ndarray | None = None,
	) -> np.ndarray:
		"""Total relative deformation of the cable."""

	@abstractmethod
	def epsilon_mecha(
		self, max_stress: np.ndarray | None = None
	) -> np.ndarray:
		"""Mechanical part of the relative deformation  of the cable."""


class LinearDeformation(Deformation):
	"""This model assumes that mechanical deformation is linear with tension."""

	def epsilon_mecha(
		self, max_stress: np.ndarray | None = None
	) -> np.ndarray:
		T_mean = self.tension_mean
		E = self.cable_array.data["young_modulus"].to_numpy()
		S = self.cable_array.data["section"].to_numpy()
		return T_mean / (E * S)

	def epsilon(
		self, current_temperature, max_stress: np.ndarray | None = None
	):
		return self.epsilon_mecha() + self.epsilon_therm(current_temperature)


class PolynomialDeformation(Deformation):
	def epsilon_mecha(
		self, max_stress: np.ndarray | None = None
	) -> np.ndarray:
		if max_stress is None:
			max_stress = np.full(self.tension_mean.shape, 0)

		T_mean = self.tension_mean
		S = self.cable_array.data["section"].to_numpy()
		E = self.cable_array.data["young_modulus"].to_numpy()
		sigma = T_mean / S

		epsilon_plastic = self.epsilon_plastic(max_stress)
		return epsilon_plastic + sigma / E

	def epsilon_plastic(self, max_stress: np.ndarray) -> np.ndarray:
		T_mean = self.tension_mean
		S = self.cable_array.data["section"].to_numpy()
		E = self.cable_array.data["young_modulus"].to_numpy()
		sigma = T_mean / S

		# if sigma is the highest constraint, the solution is the root(sigma)
		# else, the solution is on the line based on max_stress

		highest_constraint = np.fmax(sigma, max_stress)
		equation_solution = self.resolve_stress_strain_equation(
			highest_constraint
		)
		equation_solution -= highest_constraint / E
		return equation_solution

	def resolve_stress_strain_equation(self, sigma: np.ndarray) -> np.ndarray:
		"""Solves $\sigma = Polynomial(\epsilon)$"""
		T_mean = self.tension_mean
		polynom_array = np.full(
			T_mean.shape, self.cable_array.stress_strain_polynomial
		)
		poly_to_resolve = polynom_array - sigma
		return self.find_real_positive_roots(poly_to_resolve)

	@staticmethod
	def find_real_positive_roots(poly_to_resolve: np.ndarray) -> np.ndarray:
		# Can cause performance issues
		all_roots = [poly.roots() for poly in poly_to_resolve]

		all_roots_stacked = np.stack(all_roots)
		keep_solution_condition = np.logical_and(
			abs(all_roots_stacked.imag) < IMAGINARY_THRESHOLD,
			0.0 <= all_roots_stacked,
		)
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
		max_stress: np.ndarray | None = None,
	):
		return self.epsilon_mecha(max_stress) + self.epsilon_therm(
			current_temperature
		)
