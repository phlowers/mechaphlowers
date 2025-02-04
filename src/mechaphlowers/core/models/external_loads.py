# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from math import pi

import numpy as np

from mechaphlowers.entities.arrays import CableArray, WeatherArray

DEFAULT_ICE_DENSITY = 6_000


class CableLoads:
	def __init__(
		self,
		cable: CableArray,
		weather: WeatherArray,
		ice_density: float = DEFAULT_ICE_DENSITY,
	) -> None:
		self.cable = cable
		self.weather = weather
		self.ice_density = ice_density

	@property
	def load_angle(self) -> np.ndarray:
		"""Load angle (in radians)

		Returns:
			np.ndarray: load angle (beta) for each span
		"""
		linear_weight = self.cable.data.linear_weight
		ice_load = self.ice_load
		wind_load = self.wind_load

		return np.arctan(wind_load / (ice_load + linear_weight))

	@property
	def resulting_norm(
		self,
	) -> np.ndarray:
		"""Norm of the force (R) applied on the cable due to weather loads and cable own weight, per meter cable"""

		linear_weight = self.cable.data.linear_weight
		ice_load = self.ice_load
		wind_load = self.wind_load

		return np.sqrt((ice_load + linear_weight) ** 2 + wind_load**2)

	@property
	def load_coefficient(self) -> np.ndarray:
		linear_weight = self.cable.data.linear_weight
		return self.resulting_norm / linear_weight

	@property
	def ice_load(self) -> np.ndarray:
		"""Linear weight of the ice on the cable

		Returns:
			np.ndarray: linear weight of the ice for each span
		"""
		e = self.weather.data.ice_thickness.to_numpy()
		D = self.cable.data.diameter.to_numpy()
		return self.ice_density * pi * e * (e + D)

	@property
	def wind_load(self) -> np.ndarray:
		"""Linear force applied on the cable by the wind.

		Returns:
			np.ndarray: linear force applied on the cable by the wind
		"""
		P_w = self.weather.data.wind_pressure.to_numpy()
		D = self.cable.data.diameter.to_numpy()
		e = self.weather.data.ice_thickness.to_numpy()
		return P_w * (D + 2 * e)
