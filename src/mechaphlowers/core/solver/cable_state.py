# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from typing import Type

import numpy as np
from scipy import optimize  # type: ignore

from mechaphlowers.core.models.cable.deformation import (
	DeformationRTE,
	IDeformation,
)
from mechaphlowers.core.models.cable.span import (
	CatenarySpan,
	Span,
)
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import (
	CableArray,
	SectionArray,
	WeatherArray,
)


class SagTensionSolver:
	"""This class reprensents the sag tension calculation.
	It computes the horizontal tension of a cable after a change of parameters
	"""

	_ZETA = 10

	def __init__(
		self,
		section_array: SectionArray,
		cable_array: CableArray,
		weather_array: WeatherArray,
		L_ref: np.ndarray,
		span_model: Type[Span] = CatenarySpan,
		deformation_model: Type[IDeformation] = DeformationRTE,
	) -> None:
		self.a = section_array.data.span_length.to_numpy()
		self.b = section_array.data.elevation_difference.to_numpy()
		self.p = section_array.data.sagging_parameter.to_numpy()
		self.sagging_temperature = (
			section_array.data.sagging_temperature.to_numpy()
		)
		self.linear_weight = cable_array.data.linear_weight.to_numpy()
		self.E = cable_array.data.young_modulus.to_numpy()
		self.S = cable_array.data.section.to_numpy()
		self.alpha = cable_array.data.dilatation_coefficient.to_numpy()
		self.stress_strain_polynomial = cable_array.stress_strain_polynomial
		self.theta_ref = cable_array.data.temperature_reference.to_numpy()
		self.cable_loads = CableLoads(cable_array, weather_array)
		self.L_ref = L_ref
		self.span_model = span_model
		self.deformation_model = deformation_model
		self.T_h_after_change: np.ndarray | None = None

	def change_state(
		self,
		weather_array: WeatherArray,
		temp: np.ndarray,
		solver: str = "newton",
	) -> None:
		"""Method that solves the finds the new horizontal tension after a change of parameters.
		The equation to solve is : $\\delta(T_h) = O$
		Args:
			weather_array (WeatherArray): data on wind and ice
			temp (np.ndarray): current temperature
			solver (str, optional): resolution method of the equation. Defaults to "newton", which is the only method implemented for now.
		"""

		solver_dict = {"newton": optimize.newton}
		try:
			solver_method = solver_dict[solver]
		except KeyError:
			raise ValueError(f"Incorrect solver name: {solver}")
		self.cable_loads.weather = weather_array
		m = self.cable_loads.load_coefficient
		T_h0 = self.span_model.compute_T_h(self.p, m, self.linear_weight)

		# TODO adapt code to use other solving methods

		solver_result = solver_method(
			self._delta,
			T_h0,
			fprime=self._delta_prime,
			args=(m, temp, self.L_ref),
			tol=1e-5,
			full_output=True,
		)
		if not solver_result.converged.all():
			raise ValueError("Solver did not converge")
		self.T_h_after_change = solver_result.root

	def _delta(self, T_h, m, temp, L_ref):
		"""Function to solve.
		This function is the difference between two ways to compute epsilon.
		Therefore, its value should be zero.
		$\\delta = \\varepsilon_{L} - \\varepsilon_{T}$
		"""
		p = self.span_model.compute_p(T_h, m, self.linear_weight)
		L = self.span_model.compute_L(self.a, self.b, p)

		T_mean = self.span_model.compute_T_mean(self.a, self.b, p, T_h)
		epsilon_total = self.deformation_model.compute_epsilon_mecha(
			T_mean, self.E, self.S, self.stress_strain_polynomial
		) + self.deformation_model.compute_epsilon_therm(
			temp, self.theta_ref, self.alpha
		)

		return (L / L_ref - 1) - epsilon_total

	def _delta_prime(
		self,
		Th,
		m,
		temp,
		L_ref,
	):
		"""Approximation of the derivative of the function to solve
		$$\\delta'(T_h) = \\frac{\\delta(T_h + \\zeta) - \\delta(T_h)}{\\zeta}$$
		"""
		kwargs = {
			"m": m,
			"temp": temp,
			"L_ref": L_ref,
		}
		return (
			self._delta(Th + self._ZETA, **kwargs) - self._delta(Th, **kwargs)
		) / self._ZETA

	def p_after_change(self):
		"""Compute the new value of the sagging parameter after sag tension calculation"""
		m = self.cable_loads.load_coefficient
		if self.T_h_after_change is None:
			raise ValueError(
				"method change_state has to be run before calling this method"
			)
		return self.span_model.compute_p(
			self.T_h_after_change, m, self.linear_weight
		)

	def L_after_change(self):
		"""Compute the new value of the length of the cable after sag tension calculation"""
		p = self.p_after_change()
		if self.T_h_after_change is None:
			raise ValueError(
				"method change_state has to be run before calling this method"
			)
		return self.span_model.compute_L(self.a, self.b, p)
