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

	@abstractmethod
	def epsilon_therm(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Thermal part of the relative deformation of the cable, compared to a temperature_reference."""

	@abstractmethod
	def epsilon(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Total relative deformation of the cable."""

	@abstractmethod
	def epsilon_mecha(self) -> np.ndarray:
		"""Mechanical part of the relative deformation  of the cable."""


class LinearDeformation(Deformation):
	"""This model assumes that mechanical deformation is linear with tension."""

	def epsilon_therm(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Thermal part of the relative deformation of the cable, compared to a temperature_reference."""
		temp_ref = self.cable_array.data["temperature_reference"].to_numpy()
		alpha = self.cable_array.data["dilatation_coefficient"].to_numpy()
		return self.compute_epsilon_therm(current_temperature, temp_ref, alpha)

	def epsilon_mecha(self) -> np.ndarray:
		# T_mean(T_h)???
		T_mean = self.tension_mean
		E = self.cable_array.data["young_modulus"].to_numpy()
		S = self.cable_array.data["section"].to_numpy()
		return self.compute_epsilon_mecha(T_mean, E, S)

	def epsilon(self, current_temperature):
		return self.epsilon_mecha() + self.epsilon_therm(current_temperature)

	@staticmethod
	def compute_epsilon_therm(
		theta: np.ndarray, theta_ref: np.ndarray, alpha: np.ndarray
	) -> np.ndarray:
		return (theta - theta_ref) * alpha

	@staticmethod
	def compute_epsilon_mecha(
		T_mean: np.ndarray, E: np.ndarray, S: np.ndarray
	) -> np.ndarray:
		return T_mean / (E * S)

	# @staticmethod
	# def compute_epsilon(
	# 	theta: np.ndarray,
	# 	theta_ref: np.ndarray,
	# 	alpha: np.ndarray,
	# 	T_mean: np.ndarray,
	# 	E: np.ndarray,
	# 	S: np.ndarray,
	# ) -> np.ndarray:
	# 	return LinearDeformation.compute_epsilon_mecha(
	# 		T_mean, E, S
	# 	) + LinearDeformation.compute_epsilon_therm(theta, theta_ref, alpha)
