# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
import warnings
from abc import ABC

import numpy as np
import pandas as pd
import pandera as pa
from numpy.polynomial import Polynomial as Poly
from typing_extensions import Literal, Self, Type

from mechaphlowers.config import options
from mechaphlowers.data.units import Q_
from mechaphlowers.entities.errors import DataWarning
from mechaphlowers.entities.geography import get_gps_from_arrays
from mechaphlowers.entities.schemas import (
    CableArrayInput,
    ObstacleArrayInput,
    SectionArrayInput,
    WeatherArrayInput,
)
from mechaphlowers.utils import df_to_dict

logger = logging.getLogger(__name__)


class DefaultValueWarning(Warning):
    """Warning for default values being used when not provided by user."""


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
            # input_units lists every column that might need conversion, but columns can be optional. If a column is missing, we just skip it.
            if column not in data_SI.columns:
                continue
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

    def __str__(self) -> str:
        return self._data.to_string()

    def __copy__(self) -> Self:
        return type(self)(self._data)

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}\n{self.__str__()}"


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

        if sagging_parameter is None:
            warnings.warn(
                "sagging_parameter not provided. It will be set to 5 times the equivalent span.",
                DefaultValueWarning,
            )
            self.sagging_parameter = self.equivalent_span() * 5
        else:
            self.sagging_parameter = sagging_parameter
        if sagging_temperature is None:
            self.sagging_temperature = options.data.sagging_temperature_default
        else:
            self.sagging_temperature = sagging_temperature
        self.input_units = options.input_units.section_array.copy()
        self.correct_insulator_length()
        self._angles_sense: Literal["clockwise", "anticlockwise"] = (
            "anticlockwise"
        )
        logger.debug("Section Array initialized.")

    def compute_elevation_difference(self) -> np.ndarray:
        left_support_height = self._data["conductor_attachment_altitude"]
        right_support_height = left_support_height.shift(periods=-1)
        return (right_support_height - left_support_height).to_numpy()

    def compute_ground_altitude(self) -> np.ndarray:
        """Generate ground altitude array using attachment altitude, and arbitrary a support length."""
        return (
            self._data["conductor_attachment_altitude"].to_numpy()
            - options.ground.default_support_length
        )

    def correct_insulator_length(self) -> None:
        """Correct insulator length to be at least 0.01 m to avoid numerical issues."""
        if (self._data["insulator_length"] < 0.01).any():
            warnings.warn(
                "Some insulator_length values are less than 0.01 m. They will be set to 0.01 m to avoid numerical issues.",
                category=DataWarning,
            )
        self._data["insulator_length"] = self._data["insulator_length"].apply(
            lambda x: max(x, 0.01)
        )

    @property
    def angles_sense(self) -> Literal["clockwise", "anticlockwise"]:
        """Affects line_angle, crossarm_length sign

        If "anticlockwise", line_angle is anticlockwise and crossarm_length is away from user (left).
        If "clockwise", line_angle is clockwise and crossarm_length is towards user (right).

        Defaults to "anticlockwise"."""
        return self._angles_sense

    @angles_sense.setter
    def angles_sense(
        self, value: Literal["clockwise", "anticlockwise"]
    ) -> None:
        if value not in ["clockwise", "anticlockwise"]:
            raise ValueError(
                f"angles_sense should be 'clockwise' or 'anticlockwise', received {value}"
            )
        self._angles_sense = value

    @property
    def data(self) -> pd.DataFrame:
        self.correct_insulator_length()
        data_output = super().data
        data_output["insulator_weight"] = (
            Q_(data_output["insulator_mass"].to_numpy(), "kg").to("N").m
        )
        if "load_mass" in data_output:
            data_output["load_weight"] = (
                Q_(data_output["load_mass"].to_numpy(), "kg").to("N").m
            )

        self.validate_ground_altitude(data_output)
        data_output = self._adjust_angle_sense(data_output)
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

    def _adjust_angle_sense(self, data_output: pd.DataFrame) -> pd.DataFrame:
        if self.angles_sense == "clockwise":
            # use data_output instead of self._data to keep eventual unit conversion
            data_output["line_angle"] = -data_output["line_angle"]
            data_output["crossarm_length"] = -data_output["crossarm_length"]
        return data_output

    def equivalent_span(self) -> float:
        """equivalent_span

        compute equivalent span:
           $L_{eq} = \\sqrt{\\sum(L_i ^ 3)/\\sum L_i}$


        Returns:
            float: equivalent span length (m)
        """
        span_length = self._data.span_length.to_numpy()
        span_length_3 = span_length.copy() ** 3

        return np.sqrt(np.nansum(span_length_3) / np.nansum(span_length))

    def validate_ground_altitude(self, data_output: pd.DataFrame):
        if "ground_altitude" not in data_output:
            data_output["ground_altitude"] = self.compute_ground_altitude()
        else:
            ground_alt = data_output["ground_altitude"].to_numpy()
            attachment_alt = data_output[
                "conductor_attachment_altitude"
            ].to_numpy()
            wrong_ground_altitude = attachment_alt < ground_alt
            if wrong_ground_altitude.any():
                data_output["ground_altitude"] = np.where(
                    wrong_ground_altitude,
                    self.compute_ground_altitude(),
                    ground_alt,
                )
                warning_string = (
                    "ground_altitude is higher than conductor_attachment_altitude. \n"
                    f"ground_altitude being replaced by default value for incorrect supports: \n {data_output['ground_altitude'].to_numpy()}"
                )
                warnings.warn(warning_string)
                logger.warning(warning_string)

    def compute_gps_coordinates(
        self,
        start_latitude: float,
        start_longitude: float,
        start_azimuth: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GPS coordinates for the cable array.

        Args:
            start_latitude (float): Latitude of the first support in degrees.
            start_longitude (float): Longitude of the first support in degrees.
            start_azimuth (float): Azimuth of the first span in degrees, anti-clockwise. 0 means North, 90 means West.

        Returns:
            tuple[np.ndarray, np.ndarray]: Two arrays of GPS coordinates (latitude, longitude) in degrees.
        """
        line_angle_geo_degrees = (
            Q_(self.data["line_angle"].to_numpy(), "rad").to("deg").m
        )
        return get_gps_from_arrays(
            start_latitude,
            start_longitude,
            start_azimuth,
            line_angle_geo_degrees,
            self.data["span_length"].to_numpy(),
        )

    def __copy__(self) -> Self:
        copy_obj = super().__copy__()
        copy_obj.sagging_parameter = self.sagging_parameter
        copy_obj.sagging_temperature = self.sagging_temperature
        return copy_obj


class CableArray(ElementArray):
    """Physical description of a cable.

    Holds catalog data for one cable type and provides RRTS (Residual Rated
    Tensile Strength) calculations via [`rrts`][mechaphlowers.entities.arrays.CableArray.rrts]
    and [`utilization_rate`][mechaphlowers.entities.arrays.CableArray.utilization_rate].
    Use [`cut_strands`][mechaphlowers.entities.arrays.CableArray.cut_strands] to declare the number of damaged strands per layer.

    Args:
        data: Input data as a DataFrame matching [`CableArrayInput`][mechaphlowers.entities.schemas.CableArrayInput].
    """

    array_input_type: Type[pa.DataFrameModel] = CableArrayInput
    _RTS_LAYERS: list[str] = [
        "rts_layer_1",
        "rts_layer_2",
        "rts_layer_3",
        "rts_layer_4",
        "rts_layer_5",
        "rts_layer_6",
        "rts_layer_7",
        "rts_layer_8",
    ]

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
        "diameter_heart": "m",
        "section_conductor": "m^2",
        "section_heart": "m^2",
        "electric_resistance_20": "ohm.m**-1",
        "linear_resistance_temperature_coef": "K**-1",
        "radial_thermal_conductivity": "W.m**-1.K**-1",
        "rts_cable": "N",
        "rts_layer_1": "N",
        "rts_layer_2": "N",
        "rts_layer_3": "N",
        "rts_layer_4": "N",
        "rts_layer_5": "N",
        "rts_layer_6": "N",
        "rts_layer_7": "N",
        "rts_layer_8": "N",
    }
    mecha_attributes = [
        "section",
        "diameter",
        "linear_weight",
        "young_modulus",
        "dilatation_coefficient",
        "temperature_reference",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "b0",
        "b1",
        "b2",
        "b3",
        "b4",
        "diameter_heart",
        "section_heart",
        "section_conductor",
        "is_polynomial",
    ]

    thermal_attributes = [
        "diameter",
        "linear_weight",
        "diameter_heart",
        "section_heart",
        "section_conductor",
        "radial_thermal_conductivity",
        "solar_absorption",
        "emissivity",
        "electric_resistance_20",
        "linear_resistance_temperature_coef",
        "has_magnetic_heart",
    ]

    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        super().__init__(data)
        self.input_units: dict[str, str] = (
            options.input_units.cable_array.copy()
        )
        self._cut_strands: np.ndarray = np.zeros(8, dtype=int)

    @property
    def data(self) -> pd.DataFrame:
        data_output = super().data
        # add new column using linear_mass data: linear_weight
        data_output["linear_weight"] = (
            Q_(data_output["linear_mass"].to_numpy(), "kg").to("N").m
        )
        return data_output

    @property
    def data_mecha(self) -> pd.DataFrame:
        """Returns mechanical data for cable. These attributes are stored in mecha_attributes"""
        return self.data[self.mecha_attributes]

    @property
    def data_thermal(self) -> pd.DataFrame:
        """Returns thermal data for cable. These attributes are stored in thermal_attributes"""
        return self.data[self.thermal_attributes]

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

    @property
    def cut_strands(self) -> np.ndarray:
        """Number of cut strands per layer, as an 8-element integer array.

        Index 0 = layer 1, …, index 7 = layer 8.
        Defaults to all zeros until explicitly set.
        """
        return self._cut_strands

    @cut_strands.setter
    def cut_strands(self, cut_strands: list[int] | np.ndarray) -> None:
        """Set the number of cut strands per layer.

        Args:
            cut_strands: Sequence of up to 8 integers where index 0 = layer 1,
                index 1 = layer 2, …, index 7 = layer 8.

        Raises:
            ValueError: if more than 8 elements are provided.
        """
        cut_strands_arr = np.asarray(cut_strands, dtype=int)
        if len(cut_strands_arr) > 8:
            raise ValueError(
                f"cut_strands must have at most 8 elements, got {len(cut_strands_arr)}."
            )
        padded = np.zeros(8, dtype=int)
        padded[: len(cut_strands_arr)] = cut_strands_arr
        self._cut_strands = padded

    @property
    def rrts(self) -> float:
        """Residual Rated Tensile Strength (RRTS) in N.

        RRTS = rts_cable - sum(cut_strands[i] * rts_layer{i+1} for i in 0..7)

        Defaults to rts_cable when no cut strands have been set (all zeros).

        Warning: rrts is designed to act globally for the section. If one strand is cut on a span, the whole section gets the same reduced RRTS, even if the cut strand is not on this span.

        Raises:
            ValueError: if cut_strands[i] > 0 but the corresponding rts layer
                column is missing or NaN in the catalog data.
        """
        cable_data = self.data
        rts_cable = float(cable_data["rts_cable"].iloc[0])

        # Build RTS-per-layer vector (NaN where column is absent or NaN)
        rts_layers = np.array(
            [
                float(cable_data[col].iloc[0])
                if col in cable_data.columns
                and not pd.isna(cable_data[col].iloc[0])
                else np.nan
                for col in self._RTS_LAYERS
            ]
        )

        # Validate: no cut strands allowed on missing/NaN layers
        invalid_mask = (self._cut_strands > 0) & np.isnan(rts_layers)
        if invalid_mask.any():
            details = "; ".join(
                f"cut_strands[{i}] = {self._cut_strands[i]} but '{self._RTS_LAYERS[i]}' is missing or NaN"
                for i in np.where(invalid_mask)[0]
            )
            raise ValueError(details)

        rts_layers = np.nan_to_num(rts_layers, nan=0.0)
        return rts_cable - float(np.dot(self._cut_strands, rts_layers))

    def utilization_rate(self, tension_sup_N: np.ndarray) -> np.ndarray:
        """Utilization rate as a percentage of RTS, one value per span.

        rate (%) = tension_sup_N / (RRTS * safety_coefficient) * 100

        Args:
            tension_sup_N: Cable tensions in N, one value per span (e.g.
                tension_sup from BalanceEngine.get_data_spans()).

        Returns:
            Array of utilization rates in % of RTS, same length as tension_sup_N.
        """
        safety_coef = float(self.data["safety_coefficient"].iloc[0])
        return np.asarray(tension_sup_N) / (self.rrts * safety_coef) * 100


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


class ObstacleArray(ElementArray):
    """Obstacles-related data, such as obstacle altitude and distance from the line.

    They are typically used to compute clearance-related checks.
    """

    array_input_type: Type[pa.DataFrameModel] = ObstacleArrayInput
    target_units: dict[str, str] = {
        "x": "m",
        "y": "m",
        "z": "m",
    }

    def __init__(
        self,
        data: pd.DataFrame,
    ) -> None:
        super().__init__(data)
        # Check if points from the same obstacle have the same indices
        points_has_same_indices = data.duplicated(
            subset=['name', 'point_index']
        ).any()
        if points_has_same_indices:
            raise ValueError(
                "An obstacle have two points with the same point_index"
            )
        # Check if each group of 'name' has only one unique 'span_index'
        obstacle_has_same_span_index = (
            data.groupby('name')['span_index'].nunique().eq(1).all()
        )
        if not obstacle_has_same_span_index:
            raise ValueError(
                "All points from the same obstacle should have the same span_index"
            )

    def add_obstacle(
        self,
        name: str,
        span_index: int,
        coords: np.ndarray,
        object_type: str = "ground",
        support_reference: Literal['left', 'right'] = 'left',
        span_length: np.ndarray | None = None,
    ):
        """
        Method used for adding an obstacle to ObstacleArray

        coords format: [[x0, y0, z0], [x1, y1, z1],...]

        If support_reference == "left", span_length is required
        """

        if len(coords.shape) != 2 or coords.shape[1] != 3:
            raise TypeError(
                "coords have incorrect dimension: it should be (n x 3)"
            )
        nb_points = coords.shape[0]

        x = coords[:, 0]

        if support_reference == 'right':
            if span_length is None:
                raise TypeError(
                    "If support_reference is set to 'right', span_length is required"
                )
            x = self.reverse_x_coord(x, span_length, span_index)

        new_obstacle = pd.DataFrame(
            {
                "name": [name] * nb_points,
                "point_index": np.arange(nb_points),
                "span_index": [span_index] * nb_points,
                "x": x,
                "y": coords[:, 1],
                "z": coords[:, 2],
                "object_type": [object_type] * nb_points,
            }
        )
        self._data = pd.concat([self._data, new_obstacle], ignore_index=True)
        logger.debug(f"Obstacle {name} added")

    def reverse_x_coord(
        self, x: np.ndarray, span_length: np.ndarray, span_index
    ) -> np.ndarray:
        return span_length[span_index] - x

    def get_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.data["x"].to_numpy(),
            self.data["y"].to_numpy(),
            self.data["z"].to_numpy(),
        )

    @property
    def data(self) -> pd.DataFrame:
        data_output = super().data
        # Sort points by obstacle and index order
        data_output.sort_values(by=["name", "point_index"], inplace=True)
        return data_output
