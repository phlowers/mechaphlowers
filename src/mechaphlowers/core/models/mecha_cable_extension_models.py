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


class ElasticPolynomialExtensionModel(MechaCableExtensionModel):
	def epsilon_mecha(self) -> np.ndarray:
		# values hardcoded right now
		min = 0.0
		max = 0.01
		T_mean = self.sp_model.T_mean()
		S = self.cable_array.data["section"].to_numpy()

		poly_heart = self.cable_array.poly_heart
		poly_conductor = self.cable_array.poly_conductor
		poly_array = np.full(T_mean.shape, poly_heart + poly_conductor)

		poly_to_resolve = poly_array - T_mean / S
		# Can cause performance issues
		# use .toList() instead?
		all_roots = [poly.roots() for poly in poly_to_resolve]
		real_roots_in_range = np.array(
			[
				[
					root.real
					for root in roots_one_poly
					if (root.imag == 0 and (min <= root and root <= max))
				]
				for roots_one_poly in all_roots
			]
		)

		# assert len(real_roots_in_range) == 1, len(real_roots_in_range)  # TODO
		return real_roots_in_range.T[:, 0]


class ElasticPolynomialWithPreLoadExtensionModel(MechaCableExtensionModel):
	pass
