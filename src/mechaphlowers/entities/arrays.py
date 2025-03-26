# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
from typing import Type

import numpy as np
import pandas as pd
import pandera as pa
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.entities.schemas import (
	CableArrayInput,
	SectionArrayInput,
	WeatherArrayInput,
)


class ElementArray(ABC):
	array_input_type: Type[pa.DataFrameModel]

	def __init__(self, data: pd.DataFrame) -> None:
		_data = self._drop_extra_columns(data)
		self._data: pd.DataFrame = _data

	def _drop_extra_columns(self, input_data: pd.DataFrame) -> pd.DataFrame:
		"""Return a copy of the input pdt.DataFrame, without irrelevant columns.

		Note: This has no impact on the input pdt.DataFrame.
		"""
		# We need to convert Model into Schema because the strict attribute doesn't exist for Model
		array_input_schema = self.array_input_type.to_schema()
		array_input_schema.strict = 'filter'
		return array_input_schema.validate(input_data, lazy=True)

	def __str__(self) -> str:
		return self._data.to_string()

	def __copy__(self):
		return type(self)(self._data)

	@property
	@abstractmethod
	def data(self) -> pd.DataFrame:
		"""Dataframe with updated data: SI units and added columns"""

	@property
	def data_original(self) -> pd.DataFrame:
		"""Original dataframe with the exact same data as input:
		original units and no columns added
		"""
		return self._data


class SectionArray(ElementArray):
	"""Description of an overhead line section.

	Args:
	    data: Input data
	    sagging_parameter: Sagging parameter
	    sagging_temperature: Sagging temperature, in Celsius degrees
	"""

	array_input_type: Type[pa.DataFrameModel] = SectionArrayInput

	def __init__(
		self,
		data: pd.DataFrame,
		sagging_parameter: float | None = None,
		sagging_temperature: float | None = None,
	) -> None:
		super().__init__(data)  # type: ignore[arg-type]
		self.sagging_parameter = sagging_parameter
		self.sagging_temperature = sagging_temperature

	def compute_elevation_difference(self) -> np.ndarray:
		left_support_height = (
			self._data["conductor_attachment_altitude"]
			- self._data["insulator_length"]
		)
		right_support_height = left_support_height.shift(periods=-1)
		return (right_support_height - left_support_height).to_numpy()

	@property
	def data(self) -> pd.DataFrame:
		if self.sagging_parameter is None or self.sagging_temperature is None:
			raise AttributeError(
				"Cannot return data: sagging_parameter and sagging_temperature are needed"
			)
		else:
			return self._data.assign(
				elevation_difference=self.compute_elevation_difference(),
				sagging_parameter=self.sagging_parameter,
				sagging_temperature=self.sagging_temperature,
			)

	def __copy__(self):
		copy_obj = super().__copy__()
		copy_obj.sagging_parameter = self.sagging_parameter
		copy_obj.sagging_temperature = self.sagging_temperature
		return copy_obj


class CableArray(ElementArray):
	"""Physical description of a cable.

	Args:
		data: Input data
	"""

	array_input_type: Type[pa.DataFrameModel] = CableArrayInput

	def __init__(
		self,
		data: pd.DataFrame,
	) -> None:
		super().__init__(data)  # type: ignore[arg-type]

	@property
	def data(self) -> pd.DataFrame:
		"""Returns a copy of self._data that converts values into SI units"""
		data_SI = self._data.copy()
		# section is in mm²
		data_SI["section"] *= 1e-6
		# diameter is in mm
		data_SI["diameter"] *= 1e-3
		# young_modulus is in GPa
		data_SI["young_modulus"] *= 1e9
		# dilatation_coefficient is in 10⁻⁶/°C
		data_SI["dilatation_coefficient"] *= 1e-6

		# polynomial coefficients are in GPa
		for coef in ["a0", "a1", "a2", "a3", "a4"]:
			if coef in data_SI:
				data_SI[coef] *= 1e9
		return data_SI

	# TODO: remove this?
	@property
	def stress_strain_polynomial(self) -> Poly:
		"""Converts coefficients in the dataframe into polynomial"""
		coefs_poly = self.data.iloc[0][
			["a0", "a1", "a2", "a3", "a4"]
		].to_numpy()
		return Poly(coefs_poly)


class WeatherArray(ElementArray):
	"""Weather-related data, such as wind and ice.

	They're typically used to compute weather-related loads on the cable.
	"""

	array_input_type: Type[pa.DataFrameModel] = WeatherArrayInput

	def __init__(
		self,
		data: pd.DataFrame,
	) -> None:
		super().__init__(data)  # type: ignore[arg-type]

	@property
	def data(self) -> pd.DataFrame:
		data_SI = self._data.copy()
		# ice_thickness is in cm
		data_SI["ice_thickness"] *= 1e-2
		return data_SI
