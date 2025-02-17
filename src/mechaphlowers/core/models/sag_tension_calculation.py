# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
from typing import Type

import numpy as np

# import scipy.optimize
from mechaphlowers.entities.arrays import CableArray, SectionArray


class SpanEquations(ABC):
	def __init__(
		self,
		section_array: SectionArray,
		unstressed_length: np.ndarray,
		cable_array: CableArray,
		load_coefficient: np.ndarray,
	):
		self.section_array = section_array
		self.unstressed_length = unstressed_length
		self.cable_array = cable_array
		self.load_coefficient = load_coefficient

	def function_epsilon(self, T_h) -> np.ndarray:
		""""""
		L_ref = self.unstressed_length
		L = self.function_L(T_h)
		return (L - L_ref) / L_ref

	@abstractmethod
	def function_L(self, T_h) -> np.ndarray:
		""""""

	@abstractmethod
	def function_x_m(self, T_h) -> np.ndarray:
		""""""

	def function_x_n(self, T_h) -> np.ndarray:
		a = self.section_array.data.span_length.to_numpy()

		return a + self.function_x_m(T_h)

	@abstractmethod
	def function_p(self, T_h) -> np.ndarray:
		""""""


class CatenarySpanEquations(SpanEquations):
	def function_x_m(self, T_h) -> np.ndarray:
		p = self.function_p(T_h)
		# See above for explanations about following simplifying assumptions
		a = self.section_array.data.span_length.to_numpy()
		b = self.section_array.data.elevation_difference.to_numpy()

		return -a / 2 + p * np.arcsinh(b / (2 * p * np.sinh(a / (2 * p))))

	def function_L(self, T_h) -> np.ndarray:
		""""""
		return self.function_L_m(T_h) + self.function_L_n(T_h)

	def function_L_m(self, T_h) -> np.ndarray:
		p = self.function_p(T_h)
		x_m = self.function_x_m(T_h)
		return -p * np.sinh(x_m / p)

	def function_L_n(self, T_h) -> np.ndarray:
		p = self.function_p(T_h)
		x_n = self.function_x_n(T_h)
		return p * np.sinh(x_n / p)

	def function_p(self, T_h) -> np.ndarray:
		return T_h / (
			self.load_coefficient
			* self.cable_array.data.linear_weight.to_numpy()
		)


class DeformationEquations(ABC):
	""""""
	# @abstractmethod
	# def function_epsilon(self, T_h):
	# 	""""""

	# @abstractmethod
	# def function_epsilon_therm(self, T_h):
	# 	""""""

	# @abstractmethod
	# def function_epsilon_mecha(self, T_h):
	# 	""""""

class LinearDeformationEquations(DeformationEquations):
	pass

class SagTensionCalculation(ABC):
	def __init__(
		self,
		# separate arguments instead?
		section_array: SectionArray,
		unstressed_length: np.ndarray,
		cable_array: CableArray,
		span_equations_type: Type[SpanEquations] = CatenarySpanEquations,
		deformation_equations_type: Type[DeformationEquations] = LinearDeformationEquations,
		load_coefficient: np.ndarray | None = None,
	):
		self.unstressed_length = unstressed_length
		self.cable_array = cable_array
		if load_coefficient is None:
			self.load_coefficient = np.ones_like(unstressed_length)
		else:
			self.load_coefficient = load_coefficient

		self.span_equations = span_equations_type(section_array, unstressed_length, cable_array, load_coefficient) # type: ignore[arg-type]
		self.deformation_equations = deformation_equations_type()

	@abstractmethod
	def solve(self):
		""""""


class ScipySagTensionCalculation(SagTensionCalculation):
	def solve(self):
		""""""
