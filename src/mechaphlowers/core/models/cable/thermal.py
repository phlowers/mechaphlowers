# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from numpy import typing as npt
from thermohl import solver  # type: ignore
from thermohl.power.convective_cooling import (  # type: ignore
    compute_wind_attack_angle as thermohl_compute_wind_angle,
)
from thermohl.power.rte.solar_heating import (  # type: ignore
    diffuse_and_beam_radiations,
)

from mechaphlowers.entities.arrays import CableArray
from mechaphlowers.entities.errors import (
    InvalidNebulosity,
    UncertaintyNotAvailable,
)

logger = logging.getLogger(__name__)


class ThermalResults(ABC):
    """Thermal results base class."""

    INPUT_PREFIX = "input_"

    def __init__(
        self,
        input_data: dict | pd.DataFrame,
        return_inputs: bool = True,
    ):
        inputs = self._pop_inputs(input_data)
        self.inputs = inputs if return_inputs else None
        results = self.parse_results(input_data)
        self.data = results

    @staticmethod
    @abstractmethod
    def parse_results(data: dict | pd.DataFrame) -> pd.DataFrame:
        """Parse raw thermal results into a standardized DataFrame format.

        Args:
            data (dict | pd.DataFrame): Raw thermal results as dictionary or DataFrame.

        Returns:
            pd.DataFrame: Parsed results as a pandas DataFrame.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return self.data.to_string()

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}\n{self.__str__()}"

    @classmethod
    def _input_columns(cls, df: pd.DataFrame) -> list[str]:
        return [
            column
            for column in df.columns
            if column.startswith(cls.INPUT_PREFIX)
        ]

    @classmethod
    def _input_keys(cls, data: dict) -> list[str]:
        return [key for key in data.keys() if key.startswith(cls.INPUT_PREFIX)]

    @classmethod
    def _remove_input_prefix(cls, key: str) -> str:
        return key.replace(cls.INPUT_PREFIX, "")

    @classmethod
    def _pop_inputs(cls, data: dict | pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, dict):
            input_keys = cls._input_keys(data)
            inputs = pd.DataFrame(
                {
                    cls._remove_input_prefix(key): value
                    for key, value in data.items()
                    if key in input_keys
                }
            )
            for key in input_keys:
                data.pop(key)
        else:
            input_columns = cls._input_columns(data)
            inputs = data[input_columns]
            data.drop(columns=input_columns, inplace=True)
            inputs.rename(columns=cls._remove_input_prefix, inplace=True)
        return inputs


class ThermalTransientResults(ThermalResults):
    """Thermal transient results class for transient temperature calculations."""

    @staticmethod
    def parse_results(data: dict | pd.DataFrame) -> pd.DataFrame:
        """Parse transient thermal results into a time-series DataFrame.

        Converts raw transient thermal output into a DataFrame with columns for
        time, cable ID, average temperature, surface temperature, and core temperature.

        Args:
            data (dict | pd.DataFrame): Raw transient results dictionary or DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns: time, id, average_temperature,
                surface_temperature, core_temperature.

        Raises:
            TypeError: If input is a DataFrame (only dict format is supported).
        """
        if isinstance(data, pd.DataFrame):
            raise TypeError(
                "DataFrame input not supported for transient results parsing."
            )
        input_size = data["average_temperature"].shape
        return pd.DataFrame(
            {
                "time": np.tile(data["time"], input_size[1]),
                "id": np.tile(
                    np.arange(input_size[1]), (input_size[0], 1)
                ).T.flatten(),
                "average_temperature": data["average_temperature"].T.flatten(),
                "surface_temperature": data["surface_temperature"].T.flatten(),
                "core_temperature": data["core_temperature"].T.flatten(),
            }
        )


class ThermalSteadyResults(ThermalResults):
    """Thermal steady-state results parser."""

    @staticmethod
    def parse_results(
        data: dict | pd.DataFrame,
    ) -> pd.DataFrame:
        """Parse steady-state thermal results into a DataFrame.

        Converts raw steady-state thermal output into standardized DataFrame format.
        If input is already a DataFrame, returns it as-is. Otherwise converts dict to DataFrame.

        Args:
            data: Raw steady-state results as dictionary or DataFrame.

        Returns:
            Parsed results as a pandas DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        return pd.DataFrame(data)


class SteadyIntensityResults(ThermalSteadyResults):
    """Parser for thermal steady-state intensity computation."""

    pass


class SteadyTemperatureResults(ThermalSteadyResults):
    """Parser for thermal steady-state temperature computation."""

    @property
    def uncertainty(self) -> np.ndarray:
        if "uncertainty" in self.data.columns:
            return self.data["uncertainty"].to_numpy()
        raise UncertaintyNotAvailable(
            "Uncertainty not available. It hasn't been computed.\n"
            "To compute it, pass 'return_uncertainty=True' when calling thermal_engine.steady_temperature",
        )


class SolarRadiationResults(ThermalResults):
    """Diffuse and beam radiations with their sum."""

    def __init__(self, input_data):
        self.data = self.parse_results(input_data)

    @staticmethod
    def parse_results(data: dict | pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        return pd.DataFrame(data)


class ThermalForecastArray:
    """Array for input thermal forecast parameters."""

    # thl is strange to handle time series input TODO ?
    time = np.arange(10)
    wind_speed = np.linspace(0, 5, 10)
    ambient_temp = np.linspace(15, 25, 10)
    solar_irradiance = np.linspace(0, 800, 10)


# TODO: the temperature outputs have some parameters, perhaps properties are not the best way to handle that
# TODO: add latitude/longitude/altitude/azimuth in the section array
# TODO: add weather in the weather array ?
# TODO: warning, the thermal engine is using default parameters from thl, need to mirror that in mechaphlowers / future array structure ?
# TODO: conf array for intensity / target temperature ?
# TODO: builders for ThermalEngine from array
# TODO: add unit for ThermalEngine
# TODO: verify reactivity
# TODO: plot part


def check_inputs(
    nebulosity: np.ndarray[Any, np.dtype[np.integer] | np.dtype[np.floating]],
    **kwargs: npt.NDArray[np.integer | np.floating | np.datetime64],
) -> tuple[
    dict[str, npt.NDArray[np.integer | np.floating | np.datetime64]], int
]:
    """Validate input parameters.

    Ensures all inputs are numpy arrays with the same size. Also ensures that
    nebulosities are in the right range.

    Args:
        nebulosity(np.array): Nebulosity (array of int between 0 and 8).
        **kwargs: Input parameters as numpy arrays.

    Returns:
        tuple: A tuple containing:
            - dict: Dictionary with the input numpy arrays.
            - int: The common length of all arrays.

    Raises:
        ValueError: If array inputs have incompatible sizes.
        TypeError: If any input is not a numpy array.
    """
    kwargs["nebulosity"] = nebulosity

    array_length: int = nebulosity.size

    for key, value in kwargs.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Expected numpy array for '{key}', got {type(value).__name__}."
            )

        # Track and validate the length of array inputs
        if value.size != array_length:
            raise ValueError(
                f"All array inputs must have the same length. "
                f"Expected {array_length}, got {value.size} for {key}."
            )

    check_nebulosity_range(nebulosity)

    return kwargs, array_length


def check_nebulosity_range(nebulosity: np.ndarray) -> None:
    if not np.all(
        ((0 <= nebulosity) & (nebulosity <= 8)) | np.isnan(nebulosity)
    ):
        raise InvalidNebulosity(
            "Nebulosity values must be in the range [0-8]. Invalid values found in 'nebulosity'."
        )


class ThermalEngine:
    """Thermal engine is a wrapper for cable thermal modeling."""

    available_power_model = {
        "rte": solver.rte,
    }
    available_heat_equation = {
        "3tl": solver.HeatEquationType.THREE_TEMPERATURES_LEGACY,
        "3t": solver.HeatEquationType.THREE_TEMPERATURES,
    }

    def __init__(self):
        """Initialize ThermalEngine.

        Attributes:
            power_model: The power model used for thermal calculations.
            heateq: The heat equation model used.
            dict_input: Dictionary to store input parameters.
            forecast: An instance of ThermalForecastArray for time series data.
            target_temperature: Target temperature for steady-state calculations in celsius.
        """
        self.power_model = self.available_power_model.get("rte", ValueError)
        self.heateq = self.available_heat_equation.get("3tl", ValueError)
        self.dict_input = {}
        self.forecast = ThermalForecastArray()
        self.target_temperature = 65

    def set(
        self,
        cable_array: CableArray,
        latitude: np.ndarray,
        longitude: np.ndarray,
        altitude: np.ndarray,
        azimuth: np.ndarray,
        datetime_utc: npt.NDArray[np.datetime64],
        intensity: np.ndarray,
        ambient_temp: np.ndarray,
        wind_speed: np.ndarray,
        wind_angle: np.ndarray,
        nebulosity: np.ndarray,
        solar_irradiance: np.ndarray | None = None,
    ):
        """Set input parameters for thermal calculations.

        Args:
            cable_array (CableArray): An instance of CableArray containing cable properties.
            latitude (np.ndarray): Latitude values.
            longitude (np.ndarray): Longitude values.
            altitude (np.ndarray): Altitude values.
            azimuth (np.ndarray): Azimuth values.
            datetime_utc (np.ndarray): Datetime (year is indifferent).
            intensity (np.ndarray): Current intensity values.
            ambient_temp (np.ndarray): Ambient temperature values.
            wind_speed (np.ndarray): Wind speed values in m/s
            wind_angle (np.ndarray): Wind angle values in degrees, clockwise from North.
            nebulosity (np.ndarray): Nebulosity level (ints from 0 to 8). 8 is the most clouded.
            solar_irradiance (np.ndarray | None): Solar irradiance values (optional). Defaults to None.
        """
        # Handle optional solar_irradiance - create NaN array if not provided
        if solar_irradiance is None:
            solar_irradiance = np.full_like(latitude, np.nan, dtype=np.float64)

        # Normalize and validate all input parameters
        inputs, self._len = check_inputs(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            azimuth=azimuth,
            datetime_utc=datetime_utc,
            intensity=intensity,
            ambient_temp=ambient_temp,
            wind_speed=wind_speed,
            wind_angle=wind_angle,
            nebulosity=nebulosity,
            solar_irradiance=solar_irradiance,
        )

        self.dict_input = {
            "measured_global_radiation": inputs["solar_irradiance"],
            "latitude": inputs["latitude"],
            "longitude": inputs["longitude"],
            "altitude": inputs["altitude"],
            "cable_azimuth": inputs["azimuth"],
            "datetime_utc": inputs["datetime_utc"],
            "ambient_temperature": inputs["ambient_temp"],
            "wind_speed": inputs["wind_speed"],  # wind speed (m.s**-1)
            "wind_azimuth": inputs[
                "wind_angle"
            ],  # wind angle (deg, 0 means north)
            "nebulosity": inputs["nebulosity"],
            "transit": inputs["intensity"],
            "linear_mass": np.full(
                self._len, cable_array.data.linear_mass.iloc[0]
            ),
            "core_diameter": np.full(
                self._len, cable_array.data.diameter_heart.iloc[0]
            ),
            "outer_diameter": np.full(
                self._len, cable_array.data.diameter.iloc[0]
            ),
            "core_area": np.full(
                self._len, cable_array.data.section_heart.iloc[0]
            ),
            "outer_area": np.full(
                self._len, cable_array.data.section_conductor.iloc[0]
            ),
            "radial_thermal_conductivity": np.full(
                self._len, cable_array.data.radial_thermal_conductivity.iloc[0]
            ),
            "solar_absorptivity": np.full(
                self._len, cable_array.data.solar_absorption.iloc[0]
            ),
            "emissivity": np.full(
                self._len, cable_array.data.emissivity.iloc[0]
            ),
            "linear_resistance_dc_20c": np.full(
                self._len, cable_array.data.electric_resistance_20.iloc[0]
            ),
            "temperature_coeff_linear": np.full(
                self._len,
                cable_array.data.linear_resistance_temperature_coef.iloc[0],
            ),
            "magnetic_coeff": np.full(
                self._len,
                1.006 if cable_array.data.has_magnetic_heart.iloc[0] else 1.0,
            ),
            "magnetic_coeff_per_a": np.full(
                self._len,
                0.016 if cable_array.data.has_magnetic_heart.iloc[0] else 0.0,
            ),
        }
        self._load()
        logger.debug("Thermal attribute set")

    def load(self):
        """Load or reload the thermal model, and checks the shape of the input parameters.
        Can be used if the input parameters are modified without using set()."""
        check_inputs(**self.dict_input)
        self._load()

    def _load(self):
        """Load the thermal model with the current input parameters."""
        # expected to fail if arguments are not filled
        self.thermal_model = self.power_model(
            dic=self.dict_input,
            heat_equation=self.heateq,
        )

    def steady_temperature(
        self,
        intensity: np.ndarray | None = None,
        return_uncertainty: bool = False,
        return_inputs: bool = True,
    ) -> SteadyTemperatureResults:
        """Compute steady-state temperature results.

        If return_inputs=True, input data are returned in
        result.inputs as a DataFrame.

        Returns:
            SteadyTemperatureResults: An instance containing steady-state temperature data.
        """
        logger.debug("Get steady_temperature()")
        if intensity is not None:
            self.dict_input["transit"] = intensity
            self.load()
        return SteadyTemperatureResults(
            self.thermal_model.steady_temperature(
                return_uncertainty=return_uncertainty,
            ),
            return_inputs=return_inputs,
        )

    def steady_intensity(
        self,
        target_temperature: np.ndarray | None = None,
        return_inputs: bool = True,
    ) -> SteadyIntensityResults:
        """Compute steady-state intensity results.

        If return_inputs=True, input data are returned in
        result.inputs as a DataFrame.

        Returns:
            SteadyIntensityResults: An instance containing steady-state intensity data.
        """
        if target_temperature is not None:
            self.target_temperature = target_temperature

        return SteadyIntensityResults(
            self.thermal_model.steady_intensity(
                self.target_temperature,
            ),
            return_inputs=return_inputs,
        )

    def transient_temperature(
        self,
        forecast_control: ThermalForecastArray | None = None,
        return_inputs: bool = True,
    ) -> ThermalTransientResults:
        """Compute transient temperature results.

        If return_inputs=True, input data are returned in
        result.inputs as a DataFrame.

        Returns:
            ThermalTransientResults: An instance containing time-varying temperature data.
        """
        if forecast_control is not None:
            self.forecast = forecast_control

        return ThermalTransientResults(
            self.thermal_model.transient_temperature(
                offset=self.forecast.time
            ),
            return_inputs=return_inputs,
        )

    @staticmethod
    def diffuse_and_beam_solar_radiations(
        datetime_utc: npt.NDArray[np.datetime64],
        latitude: np.ndarray,
        longitude: np.ndarray,
        nebulosity: np.ndarray,
    ) -> SolarRadiationResults:
        """Compute diffuse radiation, beam radiation and their sum.

        Returns:
            SolarRadiationResults: An instance containing the results.
        """
        inputs, _ = check_inputs(
            nebulosity=nebulosity,
            datetime_utc=datetime_utc,
            latitude=latitude,
            longitude=longitude,
        )
        diffuse_radiation, beam_radiation = diffuse_and_beam_radiations(
            inputs["datetime_utc"],
            inputs["latitude"],
            inputs["longitude"],
            inputs["nebulosity"],
        )
        df = pd.DataFrame(
            {
                "diffuse_radiation": diffuse_radiation,
                "beam_radiation": beam_radiation,
                "diffuse_plus_beam_radiation": diffuse_radiation
                + beam_radiation,
            }
        )
        return SolarRadiationResults(df)

    @property
    def wind_cable_angle(self) -> np.ndarray:
        """Compute the angle between wind and cable direction.

        Triggers ambient_wind_speed mode in models.

        Returns:
            Angle in degrees between wind direction and cable azimuth.
        """
        return self.compute_wind_attack_angle(
            self.dict_input["cable_azimuth"], self.dict_input["wind_azimuth"]
        )

    @staticmethod
    def compute_wind_attack_angle(
        cable_azimuth: np.ndarray, wind_azimuth: np.ndarray
    ) -> np.ndarray:
        """Compute the angle between wind and cable.

        Args:
            cable_azimuth (np.ndarray): azimuth of the cable, in degrees
            wind_azimuth (np.ndarray): azimuth of the wind, in degrees

        Returns:
            Angle in degrees between wind direction and cable azimuth.
        """
        return np.rad2deg(
            thermohl_compute_wind_angle(cable_azimuth, wind_azimuth),
        )

    @property
    def normal_wind_mode(self):
        """Get normal wind mode status.

        Triggers normal_wind mode in models. Not implemented yet.

        Raises:
            NotImplementedError: This feature is not yet implemented.
        """
        raise NotImplementedError

    @normal_wind_mode.setter
    def normal_wind_mode(self, value: bool):
        """Set normal wind mode status.

        Triggers normal_wind mode in models. Not implemented yet.

        Args:
            value (bool): Boolean indicating if calculus should be in normal_wind mode.

        Raises:
            TypeError: If value is not a boolean (logged as warning).
        """
        # TODO: same than no wind mode but only for angle
        try:
            if not isinstance(value, bool):
                raise TypeError
            self._normal_wind_mode = bool(value)
        except TypeError:
            logger.warning("normal_wind_mode is expected boolean")

    def __len__(self) -> int:
        """Get the length of input vectors.

        Returns:
            int: Length of input vectors.
        """
        if hasattr(self, "_len"):
            return self._len
        else:
            raise AttributeError(
                "Thermal Engine has no length, please set input parameters first."
            )

    def __str__(self) -> str:
        return f"power_model={self.power_model.__name__}, heateq={self.heateq}"

    def __repr__(self) -> str:
        """Get string representation of ThermalEngine.

        Returns:
            str: String representation of the ThermalEngine instance.
        """
        class_name = type(self).__name__
        return f"<{class_name}(power_model={self.power_model.__name__}, heateq={self.heateq})>"
