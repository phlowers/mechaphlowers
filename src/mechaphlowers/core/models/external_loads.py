# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from math import pi

import numpy as np

from mechaphlowers.entities.arrays import CableArray

LINEAR_WEIGHT = 15  # TODO: use value from CableArray
ICE_DENSITY = 6_000


class IceLoad:
	def __init__(self, cable: CableArray, ice_thickness: np.ndarray) -> None:
		self.cable = cable
		self.ice_thickness = ice_thickness

	def load_coefficient(self) -> np.ndarray:  # TODO: move to separate class?
		return self.total_value() / LINEAR_WEIGHT

	def total_value(
		self,
	) -> np.ndarray:  # TODO: move to separate class?
		"""Linear force applied on the cable, for each span

		This force is the result of the cable's own weight and
		the weight of the ice on the cable.

		Returns:
			np.ndarray:
		"""
		return self.value() + LINEAR_WEIGHT

	def value(self) -> np.ndarray:
		"""Linear weight of the ice on the cable

		Returns:
			np.ndarray: ice linear weight for each span
		"""
		e = self.ice_thickness
		D = self.cable.data.diameter
		return ICE_DENSITY * pi * e * (e + D)  # FIXME
