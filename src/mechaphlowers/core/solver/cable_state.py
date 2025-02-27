# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from typing import Type

import numpy as np
from scipy import optimize

from mechaphlowers.core.models.cable.deformation import (
	Deformation,
	LinearDeformation,
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
	_ZETA = 10

	def __init__(
		self,
		section_array: SectionArray,
		cable_array: CableArray,
		weather_array: WeatherArray,
		L_ref: np.ndarray,
		span_model: Type[Span] = CatenarySpan,
		deformation_model: Type[Deformation] = LinearDeformation,
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
		self.theta_ref = cable_array.data.temperature_reference.to_numpy()
		self.cable_loads = CableLoads(cable_array, weather_array)
		self.L_ref = L_ref
		self.span_model = span_model
		self.deformation_model = deformation_model

	# default value for temp? Inlcude temperature in weather_array?
	def change_state(self, weather_array: WeatherArray, temp: np.ndarray) -> None:
		"""_summary_

		Args:
			weather_array (WeatherArray): _description_
		"""
		self.cable_loads.weather = weather_array
		m = self.cable_loads.load_coefficient
		T_h0 = self.span_model.compute_T_h(self.p, m, self.linear_weight)

		# TODO parametrize solver method with optimize minimize
		# TODO: return np array if converged = True
		T_h_after_change = optimize.newton(
			self._delta,
			T_h0,
			fprime=self._delta_prime,
			args=(m, temp, self.L_ref),
			tol=1e-5,
			full_output=True,
		).root
		self.T_h_after_change = T_h_after_change
		return T_h_after_change

	def _delta(self, T_h, m, temp, L_ref):
		p = self.span_model.compute_p(T_h, m, self.linear_weight)
		L = self.span_model.compute_L(self.a, self.b, p)

		T_mean = self.span_model.compute_T_mean(self.a, self.b, p, T_h)
		epsilon_total = self.deformation_model.compute_epsilon_mecha(
			T_mean, self.E, self.S
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
		kwargs = {
			"m": m,
			"temp": temp,
			"L_ref": L_ref,
		}
		return (
			self._delta(Th + self._ZETA, **kwargs) - self._delta(Th, **kwargs)
		) / self._ZETA


	def p_after_change(self):
		m = self.cable_loads.load_coefficient
		return self.span_model.compute_p(self.T_h_after_change, m, self.linear_weight)


	def L_after_change(self):
		p = self.p_after_change()
		return self.span_model.compute_L(self.a, self.b, p)
	