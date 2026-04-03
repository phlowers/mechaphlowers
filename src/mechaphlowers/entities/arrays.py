# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
import warnings
from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandera as pa
from numpy.polynomial import Polynomial as Poly
from typing_extensions import Literal, Self, Type

from mechaphlowers.config import options
from mechaphlowers.data.units import Q_, convert_mass_to_weight
from mechaphlowers.entities.errors import DataWarning
from mechaphlowers.entities.geography import get_gps_from_arrays

if TYPE_CHECKING:
    from mechaphlowers.core.models.cable.cable_strength import ITensileStrength
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
        "counterweight_mass": "kg",
    }

    def __init__(
        self,
        data: pd.DataFrame,
        sagging_parameter: float | None = None,
        sagging_temperature: float | None = None,
        bundle_number: int = 1,
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
        if bundle_number < 1:
            raise ValueError(
                f"bundle_number should be a positive integer. Received: {bundle_number}"
            )
        self.bundle_number = bundle_number
        self.input_units = options.input_units.section_array.copy()
        self.correct_insulator_length()
        self._angles_sense: Literal["clockwise", "anticlockwise"] = (
            "anticlockwise"
        )
        self._rope_overlay: dict[int, float] | None = None
        self._rope_lineic_mass: float | None = None
        self._original_conductor_attachment_altitude: pd.Series | None = None
        self._original_crossarm_length: pd.Series | None = None
        self._manipulation_indices: set[int] | None = None
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

    def support_manipulation(
        self, manipulation: dict[int, dict[str, float]]
    ) -> None:
        """Apply additive offsets to support geometry.

        Modifies `conductor_attachment_altitude` and/or `crossarm_length`
        in the internal data for the specified supports.

        On first call, the original values are saved so they can be
        restored with [`reset_manipulation`][mechaphlowers.entities.arrays.SectionArray.reset_manipulation].

        For each affected support, `counterweight` is set to 0 in
        [`data`][mechaphlowers.entities.arrays.SectionArray.data].
        Unaffected supports keep their original counterweight value.

        Args:
            manipulation: Dictionary mapping support index (0-based) to
                offsets. Each value is a dict with optional keys:

                - `"y"`: added to `crossarm_length` (meters)
                - `"z"`: added to `conductor_attachment_altitude` (meters)

        Raises:
            ValueError: If a support index is out of range.
            ValueError: If an inner dict contains keys other than `"y"` or `"z"`.

        Examples:
            >>> section_array.support_manipulation({1: {"z": 2.0, "y": -1.0}})
            >>> section_array.support_manipulation({0: {"z": 0.5}, 2: {"y": 3.0}})
        """
        n_supports = len(self._data)
        allowed_keys = {"y", "z"}

        for idx, offsets in manipulation.items():
            if idx < 0 or idx >= n_supports:
                raise ValueError(
                    f"Support index {idx} is out of range [0, {n_supports - 1}]"
                )
            invalid_keys = set(offsets.keys()) - allowed_keys
            if invalid_keys:
                raise ValueError(
                    f"Invalid keys {invalid_keys} for support {idx}. Allowed keys: {allowed_keys}"
                )

        # Snapshot originals on first call
        if self._original_conductor_attachment_altitude is None:
            self._original_conductor_attachment_altitude = (
                self._data["conductor_attachment_altitude"].copy()
            )
            self._original_crossarm_length = self._data[
                "crossarm_length"
            ].copy()

        for idx, offsets in manipulation.items():
            if "z" in offsets:
                self._data.loc[idx, "conductor_attachment_altitude"] += offsets[
                    "z"
                ]
            if "y" in offsets:
                self._data.loc[idx, "crossarm_length"] += offsets["y"]

        self._manipulation_indices = (
            self._manipulation_indices or set()
        ) | set(manipulation.keys())
        logger.debug(f"Support manipulation applied: {manipulation}")

    def reset_manipulation(self) -> None:
        """Restore `conductor_attachment_altitude` and `crossarm_length` to their original values.

        Reverts all changes made by
        [`support_manipulation`][mechaphlowers.entities.arrays.SectionArray.support_manipulation].
        Does nothing if no manipulation has been applied.

        Examples:
            >>> section_array.support_manipulation({1: {"z": 5.0}})
            >>> section_array.reset_manipulation()
        """
        if self._original_conductor_attachment_altitude is None:
            logger.debug(
                "reset_manipulation called but no manipulation was applied."
            )
            return

        self._data["conductor_attachment_altitude"] = (
            self._original_conductor_attachment_altitude
        )
        self._data["crossarm_length"] = self._original_crossarm_length

        self._original_conductor_attachment_altitude = None
        self._original_crossarm_length = None
        self._manipulation_indices = None

        logger.debug("Support manipulation reset to original values.")

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

    def rope_manipulation(
        self,
        rope: dict[int, float],
        rope_lineic_mass: float | None = None,
    ) -> None:
        """Override insulator length and mass for specified supports with rope values.

        The override is applied only in the
        [`data`][mechaphlowers.entities.arrays.SectionArray.data] property;
        `_data` is never modified.
        Use [`reset_rope_manipulation`][mechaphlowers.entities.arrays.SectionArray.reset_rope_manipulation]
        to remove the overlay.

        For each affected support, `counterweight` is set to 0 in
        [`data`][mechaphlowers.entities.arrays.SectionArray.data].
        Unaffected supports keep their original counterweight value.

        Args:
            rope: Dictionary mapping support index (0-based) to rope length (meters).
                Only listed supports are affected.
            rope_lineic_mass: Linear mass of the rope in kg/m. Defaults to
                ``options.data.rope_lineic_mass_default`` (``0.01`` kg/m).

        Raises:
            ValueError: If a support index is out of range.

        Examples:
            >>> section_array.rope_manipulation({1: 4.5, 2: 3.0})
            >>> section_array.rope_manipulation({0: 2.0}, rope_lineic_mass=0.05)
        """
        n_supports = len(self._data)
        for idx in rope:
            if idx < 0 or idx >= n_supports:
                raise ValueError(
                    f"Support index {idx} is out of range [0, {n_supports - 1}]"
                )

        self._rope_overlay = rope
        self._rope_lineic_mass = (
            rope_lineic_mass
            if rope_lineic_mass is not None
            else options.data.rope_lineic_mass_default
        )
        logger.debug(f"Rope manipulation applied: {rope}")

    def reset_rope_manipulation(self) -> None:
        """Remove the rope overlay and restore original insulator values in
        [`data`][mechaphlowers.entities.arrays.SectionArray.data].

        Does nothing if no rope manipulation has been applied.

        Examples:
            >>> section_array.rope_manipulation({1: 4.5})
            >>> section_array.reset_rope_manipulation()
        """
        if self._rope_overlay is None:
            logger.debug(
                "reset_rope_manipulation called but no rope manipulation was applied."
            )
            return
        self._rope_overlay = None
        self._rope_lineic_mass = None
        logger.debug("Rope manipulation cleared.")

    @property
    def data(self) -> pd.DataFrame:
        self.correct_insulator_length()
        data_output = super().data
        if self._rope_overlay is not None:
            for idx, rope_length in self._rope_overlay.items():
                data_output.loc[idx, "insulator_length"] = rope_length
                data_output.loc[idx, "insulator_mass"] = (
                    rope_length * self._rope_lineic_mass
                )
        mass_weight_conversion = {
            "insulator_mass": "insulator_weight",
            "load_mass": "load_weight",
            "counterweight_mass": "counterweight",
        }
        self.create_column_weight(data_output, mass_weight_conversion)
        if "counterweight" in data_output.columns:
            affected: set[int] = set()
            if self._manipulation_indices is not None:
                affected |= self._manipulation_indices
            if self._rope_overlay is not None:
                affected |= set(self._rope_overlay.keys())
            for idx in affected:
                data_output.loc[idx, "counterweight"] = 0.0
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
                bundle_number=self.bundle_number,
            )

    def create_column_weight(
        self, df_output: pd.DataFrame, columns_to_convert: dict[str, str]
    ) -> None:
        for column_mass, column_weight in columns_to_convert.items():
            if column_mass in self._data:
                df_output[column_weight] = convert_mass_to_weight(
                    df_output[column_mass].to_numpy()
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
    Tensile Strength) calculations via [`rrts`][mechaphlowers.entities.arrays.CableArray.rrts] and [`utilization_rate`][mechaphlowers.entities.arrays.CableArray.utilization_rate].
    Use [`cut_strands`][mechaphlowers.entities.arrays.CableArray.cut_strands] to declare the number of damaged strands per layer.

    The tensile strength model is handled by
    [`AdditiveLayerRts`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts]
    by default, but any [`ITensileStrength`][mechaphlowers.core.models.cable.cable_strength.ITensileStrength] implementation
    can be injected via the ``tensile_strength`` constructor argument.

    Args:
        data: Input data as a DataFrame matching
            [`CableArrayInput`][mechaphlowers.entities.schemas.CableArrayInput].
        tensile_strength: Optional tensile strength model. Defaults to
            [`AdditiveLayerRts`][mechaphlowers.core.models.cable.cable_strength.AdditiveLayerRts].
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
        tensile_strength: "ITensileStrength | None" = None,
    ) -> None:
        super().__init__(data)
        self.input_units: dict[str, str] = (
            options.input_units.cable_array.copy()
        )
        if tensile_strength is None:
            from mechaphlowers.core.models.cable.cable_strength import (
                AdditiveLayerRts,
            )  # noqa: PLC0415

            self._tensile_strength: ITensileStrength = AdditiveLayerRts(
                self.data
            )
        else:
            self._tensile_strength = tensile_strength

    # ------------------------------------------------------------------
    # Delegation: tensile strength model
    # ------------------------------------------------------------------

    @property
    def high_safety(self) -> bool:
        """Delegated to the tensile strength model. See
        [`ITensileStrength.high_safety`][mechaphlowers.core.models.cable.cable_strength.ITensileStrength.high_safety].
        """
        return self._tensile_strength.high_safety

    @high_safety.setter
    def high_safety(self, value: bool) -> None:
        self._tensile_strength.high_safety = value

    @property
    def safety_coefficient(self) -> float:
        """Delegated to the tensile strength model. See
        [`ITensileStrength.safety_coefficient`][mechaphlowers.core.models.cable.cable_strength.ITensileStrength.safety_coefficient].
        """
        return self._tensile_strength.safety_coefficient

    @property
    def nb_strand_per_layer(self) -> np.ndarray:
        """Delegated to the tensile strength model. See
        [`ITensileStrength.nb_strand_per_layer`][mechaphlowers.core.models.cable.cable_strength.ITensileStrength.nb_strand_per_layer].
        """
        return self._tensile_strength.nb_strand_per_layer

    def rts_coverage(self) -> float:
        """Delegated to the tensile strength model. See
        [`ITensileStrength.rts_coverage`][mechaphlowers.core.models.cable.cable_strength.ITensileStrength.rts_coverage].
        """
        return self._tensile_strength.rts_coverage()

    @property
    def cut_strands(self) -> np.ndarray:
        """Delegated to the tensile strength model. See
        [`ITensileStrength.cut_strands`][mechaphlowers.core.models.cable.cable_strength.ITensileStrength.cut_strands].
        """
        return self._tensile_strength.cut_strands

    @cut_strands.setter
    def cut_strands(self, value: list[int] | np.ndarray) -> None:
        self._tensile_strength.cut_strands = np.asarray(value)

    @property
    def rrts(self) -> float:
        """Delegated to the tensile strength model. See
        [`ITensileStrength.rrts`][mechaphlowers.core.models.cable.cable_strength.ITensileStrength.rrts].
        """
        return self._tensile_strength.rrts

    def utilization_rate(self, tension_sup_N: np.ndarray) -> np.ndarray:
        """Delegated to the tensile strength model. See
        [`ITensileStrength.utilization_rate`][mechaphlowers.core.models.cable.cable_strength.ITensileStrength.utilization_rate].
        """
        return self._tensile_strength.utilization_rate(tension_sup_N)

    # ------------------------------------------------------------------
    # End Delegation
    # ------------------------------------------------------------------

    @property
    def data(self) -> pd.DataFrame:
        data_output = super().data
        # add new column using linear_mass data: linear_weight
        data_output["linear_weight"] = convert_mass_to_weight(
            data_output["linear_mass"].to_numpy()
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
