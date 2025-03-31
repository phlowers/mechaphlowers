# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from copy import copy
from typing import List, Type

import numpy as np
import pandas as pd
from typing_extensions import Self

from mechaphlowers.api.state import StateAccessor
from mechaphlowers.core.geometry import references

# if TYPE_CHECKING:
from mechaphlowers.core.models.cable.deformation import (
	DeformationRte,
	IDeformation,
)
from mechaphlowers.core.models.cable.span import (
	CatenarySpan,
	Span,
)
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import (
	CableArray,
	ElementArray,
	SectionArray,
	WeatherArray,
)
from mechaphlowers.plotting.plot import PlotAccessor
from mechaphlowers.utils import CachedAccessor

# This parameter has to be removed later.
# This is the default resolution for spans when exporting coordinates in get_coords
RESOLUTION: int = 7


class SectionDataFrame:
	"""SectionDataFrame object is the top api object of the library.

	Inspired from dataframe, it is designed to handle data and models.
	TODO: for the moment the initialization with SectionArray and Span is explicit.
	It is not intended to be later.
	"""

	def __init__(
		self,
		section: SectionArray,
		span_model: Type[Span] = CatenarySpan,
		deformation_model: Type[IDeformation] = DeformationRte,
	):
		self.section: SectionArray = section
		self.cable: CableArray | None = None
		self.weather: WeatherArray | None = None
		self.cable_loads: CableLoads | None = None
		self.span: Span | None = None
		self.deformation: IDeformation | None = None
		self._span_model: Type[Span] = span_model
		self._deformation_model: Type[IDeformation] = deformation_model
		self.init_span_model()

	def init_span_model(self):
		"""init_span_model method to initialize span model"""
		self.span = self._span_model(
			self.section.data.span_length.to_numpy(),
			self.section.data.elevation_difference.to_numpy(),
			self.section.data.sagging_parameter.to_numpy(),
		)

	def get_coord(self) -> np.ndarray:
		"""Get x,y,z cables coordinates

		Returns:
		    np.ndarray: x,y,z array in point format
		"""

		spans = self._span_model(
			self.section.data.span_length.to_numpy(),
			self.section.data.elevation_difference.to_numpy(),
			self.section.data.sagging_parameter.to_numpy(),
		)

		# compute x_axis
		x_cable: np.ndarray = spans.x(RESOLUTION)

		# compute z_axis
		z_cable: np.ndarray = spans.z(x_cable)

		# change frame and drop last value
		# TODO refactor in a property ?
		beta = 0
		if self.cable_loads is not None:
			# TODO: here we take the max angle of the cable.
			beta = self.cable_loads.load_angle.max() * 180 / np.pi

		x_span, y_span, z_span = references.cable2span(
			x_cable[:, :-1], z_cable[:, :-1], beta=beta
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
	def data(self) -> pd.DataFrame:
		"""data property to get the data of the SectionDataFrame object with or without cable data

		Returns:
		    pd.DataFrame: data property of the SectionDataFrame object with or without cable data
		"""
		out = self.section.data
		if self.cable is not None:
			out = pd.concat([out, self.cable.data], axis=1)
		if self.weather is not None:
			out = pd.concat([out, self.weather.data], axis=1)
		return out

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

		if not (isinstance(start_value, str) and isinstance(end_value, str)):
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

	def add_cable(self, cable: CableArray):
		"""add_cable method to add a new cable to the SectionDataFrame

		Args:
			cable (CableArray): cable to add
		"""
		self._add_array(cable, CableArray)
		# type is checked in add_array
		self.span.linear_weight = self.cable.data.linear_weight.to_numpy()  # type: ignore[union-attr]
		self.init_deformation_model()

	def add_weather(self, weather: WeatherArray):
		"""add_weather method to add a new weather to the SectionDataFrame

		Args:
			weather (WeatherArray): weather to add

		Raises:
			ValueError: if cable has not been added before weather
		"""
		self._add_array(weather, WeatherArray)
		if self.cable is None:
			raise ValueError("Cable has to be added before weather")
		# weather type is checked in add_array self.cable is tested above but mypy does not understand
		self.cable_loads = CableLoads(self.cable, self.weather)  # type: ignore[union-attr,arg-type]
		self.span.load_coefficient = self.cable_loads.load_coefficient  # type: ignore[union-attr]
		self.init_deformation_model()

	def _add_array(self, var: ElementArray, type_var: Type[ElementArray]):
		"""add_array method to add a new array to the SectionDataFrame

		Args:
		    cable (ElementArray): var to add
			type_var (Type[ElementArray]): type of the var to add

		Raises:
			TypeError: if cable is not a CableArray object
			ValueError: if cable has not the same length as the section
			KeyError: if type_var is not handled by this method
		"""

		property_map = {
			CableArray: "cable",
			SectionArray: "section",
			WeatherArray: "weather",
		}

		if not isinstance(var, type_var):
			raise TypeError(f"var has to be a {type_var.__name__} object")
		try:
			property_map[type_var]
		except KeyError:
			raise TypeError(
				f"{type_var.__name__} is not handled by this method"
				f"it should be one of the {property_map}"
			)
		# Check if the var is compatible with the section
		if var._data.shape[0] != self.section._data.shape[0]:
			raise ValueError(
				f"{type_var.__name__} has to have the same length as the section"
			)

		# Add cable to the section

		self.__setattr__(property_map[type_var], var)

	def init_deformation_model(self):
		"""initialize_deformation method to initialize deformation model"""

		# Initialize deformation model
		self.deformation = self._deformation_model(
			self.cable,
			self.span.T_mean(),
			self.span.L(),
		)
		# TODO: test if L_ref change when span_model T_mean change

	plot = CachedAccessor("plot", PlotAccessor)

	state = CachedAccessor("state", StateAccessor)

	def __copy__(self):
		return type(self)(copy(self.section), self._span_model)
