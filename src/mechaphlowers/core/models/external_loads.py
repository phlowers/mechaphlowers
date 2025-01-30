# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from math import pi

import numpy as np
from pandera.typing import pandas as pdt

from mechaphlowers.entities.arrays import CableArray
from mechaphlowers.entities.schemas import LoadResultOutput

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

	def result(self) -> pdt.DataFrame[LoadResultOutput]:
		linear_weight = self.cable.data.linear_weight

		wind_load = self.wind_load()
		ice_load = self.ice_load()

		load_angle = np.arctan(wind_load / (ice_load + linear_weight))
		total_value = self.total_value(wind_load, ice_load, linear_weight)
		load_coefficient = total_value / linear_weight

		return pdt.DataFrame[LoadResultOutput](
			{
				"load_coefficient": load_coefficient,
				"load_angle": load_angle,
			}
		)

	@staticmethod
	def load_angle(wind_load, ice_load, linear_weight) -> np.ndarray:
		"""Load angle (in radians)

		Returns:
			np.array: load angle (beta) for each span
		"""
		return np.arctan(wind_load / ice_load + linear_weight)

	@staticmethod
	def total_value(wind_load, ice_load, linear_weight) -> np.ndarray:
		"""Norm of the force (R) applied on the cable due to weather loads, per meter cable"""
		return np.sqrt((ice_load + linear_weight) ** 2 + wind_load**2)

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
