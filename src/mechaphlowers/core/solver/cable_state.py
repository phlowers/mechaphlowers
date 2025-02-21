# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
from scipy import optimize
from mechaphlowers.core.models.cable.deformation import (
	LinearDeformation,
)
from mechaphlowers.core.models.cable.span import (
	CatenarySpan,
)
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import (
	CableArray,
	SectionArray,
	WeatherArray,
)

from mechaphlowers.core.models.cable.physics import Physics

class SagTensionSolver:
	
	_ZETA = 10

	def __init__(
		self,
		section_array: SectionArray,
		cable_array: CableArray,
		weather_array: WeatherArray,
		L_ref: np.ndarray
	) -> None:
		self.a = section_array.data.span_length.to_numpy()
		self.b = section_array.data.elevation_difference.to_numpy()
		self.p = section_array.data.sagging_parameter.to_numpy()
		self.sagging_temperature = section_array.data.sagging_temperature.to_numpy()
		self.linear_weight = cable_array.data.linear_weight.to_numpy()
		self.E = cable_array.data.young_modulus.to_numpy()
		self.S = cable_array.data.section.to_numpy()
		self.alpha = cable_array.data.dilatation_coefficient.to_numpy()
		self.theta_ref = cable_array.data.temperature_reference.to_numpy()
		self.cable_loads = CableLoads(cable_array, weather_array)
		self.L_ref = L_ref
		

	def change_state(self, weather_array: WeatherArray) -> None:
		"""_summary_

		Args:
			weather_array (WeatherArray): _description_
		"""
		self.cable_loads.weather = weather_array
		m = self.cable_loads.load_coefficient.to_numpy()
		Th0 = CatenarySpan.compute_T_h(self.p, m, self.linear_weight)
		temp = self.sagging_temperature
  
		L = CatenarySpan.compute_L(self.a, self.b, self.p)
		

		# TODO: parametrize Linear Deformation 
		# TODO: L_ref as static method ?
		# L_ref = Physics.compute_L_ref(temp, L, LinearDeformation)

		# TODO parametrize solver method with optimize minimize
		return optimize.newton(
			self._delta,
			Th0,
			fprime=self._delta_prime,
			args=(
				m, temp, self.L_ref
			),
			tol=1e-5,
			full_output=True
		)
 


	def _delta(self, T_h, m, temp, L_ref):
		# TODO: give CatenarySpan as an argument in __init__
		p = CatenarySpan.compute_p(T_h, m, self.linear_weight)
		L = CatenarySpan.compute_L(self.a, self.b, p)

		T_mean = CatenarySpan.compute_T_mean(self.a, self.b, p, T_h)
		# TODO: give LinearDeformation as an argument in __init__
		epsilon_total = LinearDeformation.compute_epsilon_mecha(
			T_mean, self.E, self.S
		) + LinearDeformation.compute_epsilon_therm(
			temp, self.theta_ref, self.alpha
		)

		return (L / L_ref - 1) - epsilon_total

	def _delta_prime(
		self,
		Th ,
		m ,
		temp ,
		L_ref ,
	):
		kwargs = {
			"m": m,
			"temp": temp,
			"L_ref": L_ref,
		}
		return (self._delta(Th + self._ZETA, **kwargs) - self._delta(Th, **kwargs)) / self._ZETA

