# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from abc import ABC

import numpy as np
import pandas as pd
import pandera as pa
from numpy.polynomial import Polynomial as Poly
from typing_extensions import Self, Type

from mechaphlowers.data.units import Q_, convert_mass_to_weight
from mechaphlowers.entities.schemas import (
    CableArrayInput,
    SectionArrayInput,
    WeatherArrayInput,
)
from mechaphlowers.utils import df_to_dict

logger = logging.getLogger(__name__)


class ElementArray(ABC):
    array_input_type: Type[pa.DataFrameModel]

    # dict of target units after conversion: SI units used for computations
    target_units: dict[str, str]

    def __init__(self, data: pd.DataFrame) -> None:
        _data = self._drop_extra_columns(data)
        self._data: pd.DataFrame = _data
        # dict of default input units
        self.input_units: dict[str, str] = {}

    def _drop_extra_columns(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the input pd.DataFrame, without irrelevant columns.

        Note: This has no impact on the input pd.DataFrame.
        """
        # We need to convert Model into Schema because the strict attribute doesn't exist for Model
        array_input_schema = self.array_input_type.to_schema()
        array_input_schema.strict = 'filter'
        return array_input_schema.validate(input_data, lazy=True)

    def __str__(self) -> str:
        return self._data.to_string()

    def __copy__(self) -> Self:
        return type(self)(self._data)

    def add_units(self, input_units: dict[str, str]) -> None:
        """Add dictionary of units of the data input . This will overrides the default `input_units` dict

        `input_units` has the following format:
        ```py
        {
            "column_name_0": "unit0",
            "column_name_1": "unit1",
        }
        ```

        Args:
            input_units (dict[str, str]): dictionary of columns names and corresponding units
        """
        self.input_units.update(input_units)

    @property
    def data(self) -> pd.DataFrame:
        """Returns a copy of self._data that converts values into SI units"""
        data_SI = self._data.copy()
        for column, input_unit in self.input_units.items():
            data_SI[column] = (
                Q_(self._data[column].to_numpy(), input_unit)
                .to(self.target_units[column])
                .magnitude
            )
        return data_SI

    def to_numpy(self) -> dict:
        return df_to_dict(self.data)

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
    target_units = {
        "conductor_attachment_altitude": "m",
        "crossarm_length": "m",
        "line_angle": "rad",
        "insulator_length": "m",
        "span_length": "m",
        "insulator_mass": "kg",
        "load_mass": "kg",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        sagging_parameter: float | None = None,
        sagging_temperature: float | None = None,
    ) -> None:
        super().__init__(data)  # type: ignore[arg-type]
        self.sagging_parameter = sagging_parameter
        self.sagging_temperature = sagging_temperature
        self.input_units = {"line_angle": "deg"}
        logger.debug("Section Array initialized.")

    def compute_elevation_difference(self) -> np.ndarray:
        left_support_height = self._data["conductor_attachment_altitude"]
        right_support_height = left_support_height.shift(periods=-1)
        return (right_support_height - left_support_height).to_numpy()

    @property
    def data(self) -> pd.DataFrame:
        data_output = super().data
        data_output["insulator_weight"] = convert_mass_to_weight(
            data_output["insulator_mass"].to_numpy()
        )
        if "load_mass" in data_output:
            data_output["load_weight"] = convert_mass_to_weight(
                data_output["load_mass"].to_numpy()
            )

        if self.sagging_parameter is None or self.sagging_temperature is None:
            raise AttributeError(
                "Cannot return data: sagging_parameter and sagging_temperature are needed"
            )
        else:
            sagging_parameter = np.repeat(
                np.float64(self.sagging_parameter), data_output.shape[0]
            )
            sagging_parameter[-1] = np.nan
            return data_output.assign(
                elevation_difference=self.compute_elevation_difference(),
                sagging_parameter=sagging_parameter,
                sagging_temperature=self.sagging_temperature,
            )

    def __copy__(self) -> Self:
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
    target_units: dict[str, str] = {
        "section": "m^2",
        "diameter": "m",
        "young_modulus": "Pa",
        "linear_mass": "kg/m",
        "dilatation_coefficient": "1/K",
        "temperature_reference": "°C",
        "a0": "Pa",
        "a1": "Pa",
        "a2": "Pa",
        "a3": "Pa",
        "a4": "Pa",
        "b0": "Pa",
        "b1": "Pa",
        "b2": "Pa",
        "b3": "Pa",
        "b4": "Pa",
    }

    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        super().__init__(data)
        self.input_units: dict[str, str] = {
            "section": "mm^2",
            "diameter": "mm",
            "young_modulus": "GPa",
            "dilatation_coefficient": "1/MK",
            "a0": "GPa",
            "a1": "GPa",
            "a2": "GPa",
            "a3": "GPa",
            "a4": "GPa",
            "b0": "GPa",
            "b1": "GPa",
            "b2": "GPa",
            "b3": "GPa",
            "b4": "GPa",
        }

    @property
    def data(self) -> pd.DataFrame:
        data_output = super().data
        # add new column using linear_mass data: linear_weight
        data_output["linear_weight"] = convert_mass_to_weight(
            data_output["linear_mass"].to_numpy()
        )
        return data_output

    @property
    def polynomial_conductor(self) -> Poly:
        return Poly(
            [
                self.data.a0.iloc[0],
                self.data.a1.iloc[0],
                self.data.a2.iloc[0],
                self.data.a3.iloc[0],
                self.data.a4.iloc[0],
            ]
        )

    @property
    def polynomial_heart(self) -> Poly:
        return Poly(
            [
                self.data.b0.iloc[0],
                self.data.b1.iloc[0],
                self.data.b2.iloc[0],
                self.data.b3.iloc[0],
                self.data.b4.iloc[0],
            ]
        )


class WeatherArray(ElementArray):
    """Weather-related data, such as wind and ice.

    They're typically used to compute weather-related loads on the cable.
    """

    array_input_type: Type[pa.DataFrameModel] = WeatherArrayInput
    target_units: dict[str, str] = {
        "ice_thickness": "m",
        "wind_pressure": "Pa",
    }

    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        super().__init__(data)  # type: ignore[arg-type]
        self.input_units: dict[str, str] = {
            "ice_thickness": "cm",
        }
