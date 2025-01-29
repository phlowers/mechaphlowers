# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from math import pi

import numpy as np

from mechaphlowers.entities.arrays import CableArray

ICE_DENSITY = 6_000


class ExternalLoads:
	def __init__(
		self,
		cable: CableArray,
		ice_thickness: np.ndarray,
		wind_pressure: np.ndarray,
	) -> None:
		self.cable = cable
		self.ice_thickness = ice_thickness
		self.wind_pressure = wind_pressure

	def load_coefficient(self) -> np.ndarray:
		"""Load coefficient, accounting for external loads"""
		return self.total_load() / self.cable.data.linear_weight

	def total_load(
		self,
	) -> np.ndarray:
		"""Linear force applied on the cable, for each span

		Returns:
			np.ndarray: result of the cable's own weight and
			the weight of the ice on the cable
		"""
		return self.total_external_load() + self.cable.data.linear_weight

	def total_external_load(
		self,
	) -> np.ndarray:
		return self.ice_load() + self.wind_load()

	def ice_load(self) -> np.ndarray:
		"""Linear weight of the ice on the cable

		Returns:
			np.ndarray: linear weight of the ice for each span
		"""
		e = self.ice_thickness
		D = self.cable.data.diameter
		return ICE_DENSITY * pi * e * (e + D)

	def wind_load(self) -> np.ndarray:
		"""Linear force applied on the cable by the wind.

		Returns:
			np.ndarray: linear force applied on the cable by the wind
		"""
		P_w = self.wind_pressure
		D = self.cable.data.diameter
		e = self.ice_thickness
		return (
			P_w * (D + 2 * e)
		)  # FIXME: mypy: Incompatible return value type (got "TimedeltaSeries", expected "ndarray[Any, Any]")
		# Idea: define wind_pressure and ice_thickness in a df in ExternalInputArray
