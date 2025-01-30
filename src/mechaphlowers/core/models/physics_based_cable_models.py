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
from mechaphlowers.entities.arrays import CableArray


class PhysicsBasedCableModel(ABC):
	"""This abstract class is a base class for models to compute extensions of the cable."""

	def __init__(
		self,
		cable_array: CableArray,
		tension_mean: np.ndarray,
		cable_length: np.ndarray,
		deformation_model_type: Type[
			CableDeformationModel
		] = LinearCableDeformationModel,
	):
		self.cable_array = cable_array
		self.tension_mean = tension_mean
		self.cable_length = cable_length
		self.deformation_model = deformation_model_type(
			cable_array, tension_mean
		)

	def L_ref(self, current_temperature: np.ndarray) -> np.ndarray:
		"""Unstressed cable length, at a chosen reference temperature"""
		L = self.cable_length
		epsilon = self.deformation_model.epsilon(current_temperature)
		return L / (1 + epsilon)
