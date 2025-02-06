# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
	from mechaphlowers.api.frames import SectionDataFrame


class StateAccessor:
	"""shortcut accessor class for state calculus"""

	def __init__(self, section: SectionDataFrame):
		self.section: SectionDataFrame = section

	def L_ref(self, current_temperature: float | np.ndarray) -> np.ndarray:
		"""L_ref values for the current temperature

		Args:
			current_temperature (float | np.ndarray): current temperature in degrees Celsius

		Raises:
			ValueError: if current_temperature is not a float or an array with the same length as the section

		Returns:
			np.ndarray: L_ref values
		"""
		if self.section.physics is None:
			raise ValueError(
				"Physics model is not defined: setting cable usually sets physics model"
			)
		if isinstance(current_temperature, (float, int)):
			current_temperature = np.full(
				self.section.section.data.shape[0], float(current_temperature)
			)
		if not isinstance(current_temperature, np.ndarray):
			raise ValueError(
				"Current temperature should be a float or an array"
			)
		if isinstance(current_temperature, np.ndarray):
			if (
				current_temperature.shape[0]
				!= self.section.section.data.shape[0]
			):
				raise ValueError(
					"Current temperature should have the same length as the section"
				)
		return self.section.physics.L_ref(current_temperature)
