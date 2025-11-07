# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from copy import copy
from typing import Callable, List, Type

import numpy as np
import pandas as pd
from typing_extensions import Self

from mechaphlowers.api.state import StateAccessor
from mechaphlowers.config import options

# if TYPE_CHECKING:
from mechaphlowers.core.models.cable.deformation import (
    DeformationRte,
    IDeformation,
)
from mechaphlowers.core.models.cable.span import (
    CatenarySpan,
    ISpan,
)
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import (
    CableArray,
    ElementArray,
    SectionArray,
    WeatherArray,
)
from mechaphlowers.entities.data_container import DataContainer
from mechaphlowers.plotting.plot import PlotAccessor
from mechaphlowers.utils import CachedAccessor, df_to_dict

logger = logging.getLogger(__name__)


map_model_options: dict[str, Callable] = {
    "CatenarySpan": CatenarySpan,
    "DeformationRte": DeformationRte,
}


class SectionDataFrame:
    """SectionDataFrame object is the top api object of the library.

    Inspired from dataframe, it is designed to handle data and models.
    TODO: for the moment the initialization with SectionArray and Span is explicit.
    It is not intended to be later.
    """

    def __init__(
        self,
        section_array: SectionArray,
    ):
        # Assign
        self.section_array: SectionArray = section_array

        # Declare
        self.cable: CableArray | None = None
        self.weather: WeatherArray | None = None
        self.data_container: DataContainer = DataContainer()
        self.cable_loads: CableLoads | None = None
        self.span: ISpan
        self.deformation: IDeformation | None = None
        self._span_model: Type[ISpan]
        self._deformation_model: Type[IDeformation]

        # Initialize
        self.data_container.add_section_array(section_array)
        self.init_type_model()
        self.init_span_model()

    def init_type_model(
        self,
        span_model: Type[ISpan] | None = None,
        deformation_model: Type[IDeformation] | None = None,
    ):
        """init_type_model method to initialize type model"""
        if span_model is None:
            try:
                self._span_model = map_model_options[
                    options.compute.span_model
                ]  # type: ignore[no-redef,assignment]
            except KeyError:
                raise options.OptionError(
                    f"Model {options.compute.span_model=} is not available."
                )
        else:
            self._span_model = span_model  # type: ignore[no-redef,assignment]
        if deformation_model is None:
            try:
                self._deformation_model = map_model_options[
                    options.compute.deformation_model
                ]  # type: ignore[no-redef, assignment]
            except KeyError:
                raise options.OptionError(
                    f"Model {options.compute.deformation_model=} is not available."
                )
        else:
            self._deformation_model = deformation_model  # type: ignore[no-redef,assignment]

    def init_span_model(self) -> None:
        """init_span_model method to initialize span model"""
        self.span = self._span_model(**self.data_container.__dict__)

    @property
    def data(self) -> pd.DataFrame:
        """data property to get the data of the SectionDataFrame object with or without cable data

        Returns:
            pd.DataFrame: data property of the SectionDataFrame object with or without cable data
        """
        out = self.section_array.data
        if self.cable is not None:
            # repeat to adjust size: CableArray only has one row
            cable_data_repeat = self.cable.data.loc[
                np.repeat(self.cable.data.index, out.shape[0])
            ].reset_index(drop=True)
            out = pd.concat([out, cable_data_repeat], axis=1)
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
            raise ValueError(f"{len(between)=} argument is expected to be 2")

        start_value: str = between[0]
        end_value: str = between[1]

        if not (isinstance(start_value, str) and isinstance(end_value, str)):
            raise TypeError(
                "Strings are expected for support name inside the between list argument"
            )

        if start_value == end_value:
            raise ValueError("At least two rows has to be selected")

        if int(self.data["name"].isin(between).sum()) != 2:
            raise ValueError(
                "One of the two name given in the between argument are not existing"
            )

        return_sf = copy(self)
        return_sf.data.set_index("name").loc[start_value, :].index

        idx_start = (
            return_sf.data.loc[return_sf.data["name"] == start_value, :]
            .index[0]
            .item()
        )
        idx_end = (
            return_sf.data.loc[return_sf.data["name"] == end_value, :]
            .index[0]
            .item()
        )

        if idx_end <= idx_start:
            raise ValueError("First selected item is after the second one")

        return_sf.section_array._data = return_sf.section_array._data.iloc[
            idx_start : idx_end + 1
        ]
        if return_sf.weather is not None:
            return_sf.weather._data = return_sf.weather._data.iloc[
                idx_start : idx_end + 1
            ]
        return_sf.update()
        return return_sf

    def add_cable(self, cable: CableArray):
        """add_cable method to add a new cable to the SectionDataFrame

        Args:
                cable (CableArray): cable to add
        """
        self._add_array(cable, CableArray)
        # type is checked in add_array
        self.data_container.add_cable_array(cable)
        self.update_cable()
        self.init_deformation_model()

    def update_cable(self) -> None:
        """update_cable method to update the cable-related properties"""
        self.span.linear_weight = self.cable.data.linear_weight.iloc[0]  # type: ignore[union-attr]

    def add_weather(self, weather: WeatherArray) -> None:
        """add_weather method to add a new weather to the SectionDataFrame

        Args:
                weather (WeatherArray): weather to add

        Raises:
                ValueError: if cable has not been added before weather
        """
        self._add_array(weather, WeatherArray)
        # Check if the var is compatible with the section_array
        if weather._data.shape[0] != self.section_array._data.shape[0]:
            raise ValueError(
                "WeatherArray has to have the same length as the section"
            )
        if self.cable is None:
            raise ValueError("Cable has to be added before weather")
        # weather type is checked in add_array self.cable is tested above but mypy does not understand
        self.data_container.add_weather_array(weather)
        self.update_weather()

    def update_weather(self) -> None:
        """update_weather method to update the weather-related properties"""
        self.cable_loads = CableLoads(**self.data_container.__dict__)  # type: ignore[union-attr,arg-type]
        self.span.load_coefficient = self.cable_loads.load_coefficient  # type: ignore[union-attr]
        # Run change state solver in order to update sagging parameter
        self.state.change(
            self.data_container.sagging_temperature, self.weather
        )
        # Data format is unusual:
        # self.weather is used as argument, but using data_container instead should be better
        self.init_deformation_model()

    # How to manage case where type_var = SectionArray
    def _add_array(self, var: ElementArray, type_var: Type[ElementArray]):
        """add_array method to add a new array to the SectionDataFrame

        Args:
            cable (ElementArray): var to add
                type_var (Type[ElementArray]): type of the var to add

        Raises:
                TypeError: if cable is not a CableArray object
                ValueError: if cable has not the same length as the section_array
                KeyError: if type_var is not handled by this method
        """

        property_map = {
            CableArray: "cable",
            SectionArray: "section_array",
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

        # Add array to the SectionDataFrame
        self.__setattr__(property_map[type_var], var)

    def init_deformation_model(self) -> None:
        """initialize_deformation method to initialize deformation model"""
        if self.cable is None:
            raise ValueError("Cable has to be added before deformation model")
        # Initialize deformation model
        self.span.compute_values()
        self.deformation = self._deformation_model(
            **self.data_container.__dict__,
            tension_mean=self.span.T_mean(),
            cable_length=self.span.L,
        )
        # TODO: test if L_ref change when span_model T_mean change

    def update(self) -> None:
        """update method to update the state of the object"""
        update_data = df_to_dict(self.section_array.data)
        if self.weather is not None:
            update_data = update_data | df_to_dict(self.weather.data)
        if self.cable is not None:
            update_data = update_data | df_to_dict(self.cable.data)

        self.data_container.update_from_dict(update_data)
        self.span.update_from_dict(self.data_container.__dict__)
        if self.cable_loads is not None:
            self.cable_loads.update_from_dict(update_data)

        if self.cable is not None:
            self.update_cable()
        if self.weather is not None:
            self.update_weather()

    plot = CachedAccessor("plot", PlotAccessor)

    state = CachedAccessor("state", StateAccessor)

    def __copy__(self) -> Self:
        out = type(self)(copy(self.section_array))
        out.cable = copy(self.cable)
        out.weather = copy(self.weather)
        out.data_container = copy(self.data_container)
        out.cable_loads = copy(self.cable_loads)
        out.span = copy(self.span)
        out.deformation = copy(self.deformation)

        out.init_type_model(
            span_model=self._span_model,
            deformation_model=self._deformation_model,
        )
        out.init_span_model()
        try:
            out.init_deformation_model()
        except ValueError:
            pass

        return out
