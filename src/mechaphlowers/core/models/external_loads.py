# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from math import pi

import numpy as np

from mechaphlowers.entities.arrays import CableArray

DEFAULT_ICE_DENSITY = 6_000


class WeatherLoads:
	def __init__(
		self,
		cable: CableArray,
		ice_thickness: np.ndarray,
		wind_pressure: np.ndarray,
		ice_density: float = DEFAULT_ICE_DENSITY,
	) -> None:
		self.cable = cable
		self.ice_thickness = ice_thickness
		self.wind_pressure = wind_pressure
		self.ice_density = ice_density

	def load_coefficient(self) -> np.ndarray:
		"""Load coefficient, accounting for external loads"""
		return self.total_with_linear_weight() / self.cable.data.linear_weight

	def total_with_linear_weight(
		self,
	) -> np.ndarray:
		"""Linear force applied on the cable, for each span

		Returns:
			np.ndarray: result of the cable's own weight and
			the external load
		"""
		return self.total_value() + self.cable.data.linear_weight

	def total_value(
		self,
	) -> np.ndarray:
		"""Linear force applied on the cable due to external loads"""
		q_wind = self.wind_load()
		q_ice = self.ice_load()
		linear_weight = self.cable.data.linear_weight
		return np.sqrt((q_ice + linear_weight) ** 2 + q_wind**2)

	def ice_load(self) -> np.ndarray:
		"""Linear weight of the ice on the cable

		Returns:
			np.ndarray: linear weight of the ice for each span
		"""
		e = self.ice_thickness
		D = self.cable.data.diameter
		return self.ice_density * pi * e * (e + D)

	def wind_load(self) -> np.ndarray:
		"""Linear force applied on the cable by the wind.

		Returns:
			np.ndarray: linear force applied on the cable by the wind
		"""
		P_w = self.wind_pressure
		D = self.cable.data.diameter
		e = self.ice_thickness
		return P_w * (D + 2 * e)

	def load_angle(self) -> np.ndarray:
		"""Load angle (in radians)

		Returns:
			np.array: load angle (beta) for each span
		"""
		# TODO: improve perf? cf. trig schema
		return np.arctan(
			self.wind_load()
			/ (self.ice_load() + self.cable.data.linear_weight)
		)
