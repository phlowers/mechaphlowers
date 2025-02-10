# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import numpy as np

from mechaphlowers.entities.arrays import CableArray


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
		constraint_max: np.ndarray | None = None,
	) -> np.ndarray:
		"""Total relative deformation of the cable."""

	@abstractmethod
	def epsilon_mecha(
		self, constraint_max: np.ndarray | None = None
	) -> np.ndarray:
		"""Mechanical part of the relative deformation  of the cable."""


class LinearDeformation(Deformation):
	"""This model assumes that mechanical deformation is linear with tension."""

	def epsilon_mecha(
		self, constraint_max: np.ndarray | None = None
	) -> np.ndarray:
		T_mean = self.tension_mean
		E = self.cable_array.data["young_modulus"].to_numpy()
		S = self.cable_array.data["section"].to_numpy()
		return T_mean / (E * S)

	def epsilon(
		self, current_temperature, constraint_max: np.ndarray | None = None
	):
		return self.epsilon_mecha() + self.epsilon_therm(current_temperature)


class PolynomialDeformation(Deformation):
	def epsilon_mecha(
		self, constraint_max: np.ndarray | None = None
	) -> np.ndarray:
		if constraint_max is None:
			constraint_max = np.full(self.tension_mean.shape, 0)

		T_mean = self.tension_mean
		S = self.cable_array.data["section"].to_numpy()
		sigma = T_mean / S

		E = self.cable_array.data["young_modulus"].to_numpy()

		# if sigma is the highest constraint, the solution is the root(sigma)
		# else, the solution is on the line based on constraint_max
		# so the solution is root(constraint_max) - sigma/E
		if (sigma >= constraint_max).all():
			equation_solution = self.find_roots_polynom(sigma)
		elif (sigma < constraint_max).all():
			epsilon_max = self.find_roots_polynom(constraint_max)
			equation_solution = epsilon_max - sigma / E
		else:
			is_lower_than_before = sigma < constraint_max
			highest_constraint = np.fmax(sigma, constraint_max)
			equation_solution = self.find_roots_polynom(highest_constraint)
			for index in range(equation_solution.size):
				if is_lower_than_before[index]:
					equation_solution[index] -= sigma[index] / E[index]
		return equation_solution

	def find_roots_polynom(self, sigma: np.ndarray) -> np.ndarray:
		"""Solves $\sigma = Polynom(\epsilon)$"""
		T_mean = self.tension_mean

		polynom_array = np.full(T_mean.shape, self.cable_array.polynom)

		poly_to_resolve = polynom_array - sigma
		# Can cause performance issues
		all_roots = [poly.roots() for poly in poly_to_resolve]

		all_roots_stacked = np.stack(all_roots)
		keep_solution_condition = np.logical_and(
			abs(all_roots_stacked.imag) < 1e-5, 0.0 <= all_roots_stacked
		)
		real_positive_roots = np.where(
			keep_solution_condition, all_roots_stacked, np.inf
		)
		real_smallest_root = real_positive_roots.min(axis=1).real
		if np.inf in real_smallest_root:
			raise ValueError("No solution found for at least one span")
		return real_smallest_root

	def epsilon(
		self,
		current_temperature: np.ndarray,
		constraint_max: np.ndarray | None = None,
	):
		return self.epsilon_mecha(constraint_max) + self.epsilon_therm(
			current_temperature
		)
