# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0from abc import ABC, abstractmethod

from abc import ABC, abstractmethod

import numpy as np
from pandera.typing import pandas as pdt

from mechaphlowers.core.models.space_position_cable_models import (
	SpacePositionCableModel,
)
from mechaphlowers.entities.arrays import CableArray, CableArrayInput


class MechaCableExtensionModel(ABC):
	"""This abstract class is a base class for models to compute extensions of the cable."""

	def __init__(
		self,
		sp_model: SpacePositionCableModel,
		data_cable: pdt.DataFrame[CableArrayInput],
	):
		self.cable_array = CableArray(data_cable)
		self.sp_model = sp_model

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

