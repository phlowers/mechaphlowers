# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from copy import copy
from typing import List, Self, Type

import numpy as np

from mechaphlowers.core.geometry import references
from mechaphlowers.core.models.cable_models import (
	CatenaryCableModel,
	SpacePositionCableModel,
)
from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.plotting.plot import PlotAccessor
from mechaphlowers.utils import CachedAccessor

# This parameter has to be removed later.
# This is the default resolution for spans when exporting coordinates in get_coords
RESOLUTION: int = 10


class SectionDataFrame:
	"""SectionDataFrame object is the top api object of the library.

	Inspired from dataframe, it is designed to handle data and models.
	TODO: for the moment the initialization with SectionArray and SpacePositionCableModel is explicit.
	It is not intended to be later.
	"""

	def __init__(
		self,
		section: SectionArray,
		span_model: Type[SpacePositionCableModel] = CatenaryCableModel,
	):
		self.section: SectionArray = section
		self.span_model: Type[SpacePositionCableModel] = span_model

	def get_coord(self) -> np.ndarray:
		"""Get x,y,z cables coordinates

		Returns:
		    np.ndarray: x,y,z array in point format
		"""

		spans = self.span_model(
			self.section.data.span_length.to_numpy(),
			self.section.data.elevation_difference.to_numpy(),
			self.section.data.sagging_parameter.to_numpy(),
		)

		# compute x_axis
		x_cable: np.ndarray = spans.x(RESOLUTION)

		# compute z_axis
		z_cable: np.ndarray = spans.z(x_cable)

		# change frame and drop last value
		x_span, y_span, z_span = references.cable2span(
			x_cable[:, :-1], z_cable[:, :-1], beta=0
		)

		altitude: np.ndarray = (
			self.section.data.conductor_attachment_altitude.to_numpy()
		)
		span_length: np.ndarray = self.section.data.span_length.to_numpy()
		crossarm_length: np.ndarray = (
			self.section.data.crossarm_length.to_numpy()
		)
		insulator_length: np.ndarray = (
			self.section.data.insulator_length.to_numpy()
		)

		# TODO: the content of this function is not generic enough. An upcoming feature will change that.
		x_span, y_span, z_span = references.translate_cable_to_support(
			x_span,
			y_span,
			z_span,
			altitude,
			span_length,
			crossarm_length,
			insulator_length,
		)

		# dont forget to flatten the arrays and stack in a 3xNpoints array
		# Ex: z_span = array([[10., 20., 30.], [11., 12. ,13.]]) -> z_span.reshape(-1) = array([10., 20., 30., 11., 12., 13.])
		return np.vstack(
			[x_span.T.reshape(-1), y_span.T.reshape(-1), z_span.T.reshape(-1)]
		).T

	@property
	def data(self):
		"""data property to return SectionArray data property

		Returns:
		    np.ndarray: SectionArray data from input
		"""
		return self.section.data

	def select(self, between: List[str]) -> Self:
		"""select enable to select a part of the line based on support names

		Args:
		    between (List[str]): list of 2 elements [start support name, end support name].
		        End name is expected to be after start name in the section order

		Raises:
		    TypeError: if between is not a list or has no string inside
		    ValueError: length(between) > 2 | names not existing or identical


		Returns:
		    Self: copy of SectionDataFrame with the selected data
		"""

		if not isinstance(between, list):
			raise TypeError()

		if len(between) != 2:
			raise ValueError("{len(between)=} argument is expected to be 2")

		start_value: str = between[0]
		end_value: str = between[1]

		if not (isinstance(start_value, str) and isinstance(start_value, str)):
			raise TypeError(
				"Strings are expected for support name inside the between list argument"
			)

		if start_value == end_value:
			raise ValueError("At least two rows has to be selected")

		if int(self.section.data["name"].isin(between).sum()) != 2:
			raise ValueError(
				"One of the two name given in the between argument are not existing"
			)

		return_sf = copy(self)
		return_sf.data.set_index("name").loc[start_value, :].index

		idx_start = (
			return_sf.data.loc[return_sf.data.name == start_value, :]
			.index[0]
			.item()
		)
		idx_end = (
			return_sf.data.loc[return_sf.data.name == end_value, :]
			.index[0]
			.item()
		)

		if idx_end <= idx_start:
			raise ValueError("First selected item is after the second one")

		return_sf.section._data = return_sf.section._data.iloc[
			idx_start : idx_end + 1
		]

		return return_sf

	plot = CachedAccessor("plot", PlotAccessor)

	def __copy__(self):
		return type(self)(copy(self.section), self.span_model)
