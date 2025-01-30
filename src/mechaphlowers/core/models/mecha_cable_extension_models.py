# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0from abc import ABC, abstractmethod

from abc import ABC, abstractmethod

import numpy as np

from mechaphlowers.core.models.space_position_cable_models import (
	SpacePositionCableModel,
)
from mechaphlowers.entities.arrays import CableArray


class MechaCableExtensionModel(ABC):
	# TODO: write docstring
	"""This abstract class is a base class for models to compute extensions of the cable."""

	def __init__(
		self,
		sp_model: SpacePositionCableModel,
		cable_array: CableArray,
	):
		self.cable_array = cable_array
		self.sp_model = sp_model

	def epsilon_therm(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Thermal part of the relative extension of the cable, compared to a temperature_reference."""
		temp_ref = self.cable_array.data["temperature_reference"].to_numpy()
		alpha = self.cable_array.data["dilatation_coefficient"].to_numpy()
		return (current_temperature - temp_ref) * alpha

	def epsilon(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Total strain of the cable."""
		return self.epsilon_mecha() + self.epsilon_therm(current_temperature)

	@abstractmethod
	def epsilon_mecha(self) -> np.ndarray:
		"""Mechanical part of the relative extension of the cable."""


class ElasticLinearExtensionModel(MechaCableExtensionModel):
	"""This model assumes that mechanical extension is linear with tension."""

	def epsilon_mecha(self) -> np.ndarray:
		T_mean = self.sp_model.T_mean()
		E = self.cable_array.data["young_modulus"].to_numpy()
		S = self.cable_array.data["section"].to_numpy()
		return T_mean / (E * S)
