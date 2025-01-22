# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0from abc import ABC, abstractmethod

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from pandera.typing import pandas as pdt

from mechaphlowers.core.models.space_position_cable_models import (
	SpacePositionCableModel,
)
from mechaphlowers.entities.arrays import CableArray, CableArrayInput


class PhysicsBasedCableModel(ABC):
	def __init__(
		self,
		sp_model: SpacePositionCableModel,
		data_cable: pdt.DataFrame[CableArrayInput] | pd.DataFrame,
	):
		self.cable_array = CableArray(data_cable)
		# Fetch linear weight given by user into SPModel
		sp_model.linear_weight = data_cable["linear_weight"].to_numpy()
		self.sp_model = sp_model

	@abstractmethod
	def epsilon_mecha(self) -> np.ndarray:
		"""Mechanical part of the relative extension of the cable."""

	# TODO: change argument type: np array -> float
	@abstractmethod
	def epsilon_therm(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Thermal part of the relative extension of the cable, compared to a temperature_reference."""

	@abstractmethod
	def epsilon(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Total strain of the cable."""

	@abstractmethod
	def L_ref(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Unstressed cable length, at a chosen reference temperature"""


class PhysicsBasedCableModelImpl(PhysicsBasedCableModel):
	def epsilon_mecha(self) -> np.ndarray:
		T_mean = self.sp_model.T_mean()
		E = self.cable_array.data["young_modulus"]
		S = self.cable_array.data["section"]
		return T_mean / (E * S)

	def epsilon_therm(self, current_temperature: np.ndarray) -> np.ndarray:
		temp_ref = self.cable_array.data["temperature_reference"]
		alpha = self.cable_array.data["dilatation_coefficient"]
		return (current_temperature - temp_ref) * alpha

	def epsilon(self, current_temperature: np.ndarray) -> np.ndarray:
		return self.epsilon_mecha() + self.epsilon_therm(current_temperature)

	def L_ref(self, current_temperature: np.ndarray) -> np.ndarray:
		L = self.sp_model.L()
		epsilon = self.epsilon(current_temperature)
		return L / (1 + epsilon)
