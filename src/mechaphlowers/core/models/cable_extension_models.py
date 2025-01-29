# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0from abc import ABC, abstractmethod

from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from mechaphlowers.core.models.mecha_cable_extension_models import (
	ElasticLinearExtensionModel,
	MechaCableExtensionModel,
)
from mechaphlowers.core.models.space_position_cable_models import (
	SpacePositionCableModel,
)
from mechaphlowers.entities.arrays import CableArray


class CableExtensionModel(ABC):
	"""This abstract class is a base class for models to compute extensions of the cable."""

	def __init__(
		self,
		sp_model: SpacePositionCableModel,
		cable_array: CableArray,
		mecha_model_type: Type[
			MechaCableExtensionModel
		] = ElasticLinearExtensionModel,
	):
		# Fetch linear weight given by user into SPModel
		self.cable_array = cable_array
		sp_model.linear_weight = cable_array.data["linear_weight"].to_numpy()
		self.sp_model = sp_model
		self.mecha_model = mecha_model_type(self.sp_model, cable_array)

	@abstractmethod
	def epsilon_therm(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Thermal part of the relative extension of the cable, compared to a temperature_reference."""

	@abstractmethod
	def epsilon(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Total strain of the cable."""

	@abstractmethod
	def L_ref(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Unstressed cable length, at a chosen reference temperature"""


class CableExtensionModelImpl(CableExtensionModel):
	"""This model assumes that mechanical extension is linear with tension."""

	def epsilon_therm(self, current_temperature: np.ndarray) -> np.ndarray:
		temp_ref = self.cable_array.data["temperature_reference"].to_numpy()
		alpha = self.cable_array.data["dilatation_coefficient"].to_numpy()
		return (current_temperature - temp_ref) * alpha

	def epsilon(self, current_temperature: np.ndarray) -> np.ndarray:
		return self.mecha_model.epsilon_mecha() + self.epsilon_therm(
			current_temperature
		)

	def L_ref(self, current_temperature: np.ndarray) -> np.ndarray:
		L = self.sp_model.L()
		epsilon = self.epsilon(current_temperature)
		return L / (1 + epsilon)
