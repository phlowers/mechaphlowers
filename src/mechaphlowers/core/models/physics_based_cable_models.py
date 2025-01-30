# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0from abc import ABC, abstractmethod

from abc import ABC
from typing import Type

import numpy as np

from mechaphlowers.core.models.cable_deformation_models import (
	CableDeformationModel,
	LinearCableDeformationModel,
)
from mechaphlowers.core.models.space_position_cable_models import (
	SpacePositionCableModel,
)
from mechaphlowers.entities.arrays import CableArray


class PhysicsBasedCableModel(ABC):
	"""This abstract class is a base class for models to compute extensions of the cable."""

	def __init__(
		self,
		sp_model: SpacePositionCableModel,
		cable_array: CableArray,
		mecha_model_type: Type[
			CableDeformationModel
		] = LinearCableDeformationModel,
	):
		# Fetch linear weight given by user into SPModel
		self.cable_array = cable_array
		sp_model.linear_weight = cable_array.data["linear_weight"].to_numpy()
		self.sp_model = sp_model
		self.mecha_model = mecha_model_type(self.sp_model, cable_array)

	def L_ref(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Unstressed cable length, at a chosen reference temperature"""
		L = self.sp_model.L()
		epsilon = self.mecha_model.epsilon(current_temperature)
		return L / (1 + epsilon)
