# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from thermohl import solver  # type: ignore
from typing_extensions import Self

from mechaphlowers.entities.arrays import CableArray

logger = logging.getLogger(__name__)


class ThermalResults(ABC):
    """Thermal results base class."""

    def __init__(self, input_data: dict | pd.DataFrame):
        self.data = self.parse_results(input_data)

    @staticmethod
    @abstractmethod
    def parse_results(data: dict | pd.DataFrame) -> pd.DataFrame:
        """Parse raw thermal results into a standardized DataFrame format.

        Args:
            data (dict | pd.DataFrame): Raw thermal results as dictionary or DataFrame.

        Returns:
            pd.DataFrame: Parsed results as a pandas DataFrame.
        """
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return self.data.to_string()

    def __copy__(self) -> Self:
        return type(self)(self.data)

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f"{class_name}\n{self.__str__()}"


class ThermalTransientResults(ThermalResults):
    """Thermal transient results class for transient temperature calculations."""

    def __init__(self, input_data: dict | pd.DataFrame):
        """Initialize transient thermal results.

        Args:
            input_data (dict | pd.DataFrame): Raw transient thermal results data.
        """
        super().__init__(input_data)

    @staticmethod
    def parse_results(data: dict | pd.DataFrame) -> pd.DataFrame:
        """Parse transient thermal results into a time-series DataFrame.

        Converts raw transient thermal output into a DataFrame with columns for
        time, cable ID, average temperature, surface temperature, and core temperature.

        Args:
            data (dict | pd.DataFrame): Raw transient results dictionary or DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns: time, id, t_avg, t_surf, t_core.

        Raises:
            TypeError: If input is a DataFrame (only dict format is supported).
        """
        if isinstance(data, pd.DataFrame):
            raise TypeError(
                "DataFrame input not supported for transient results parsing."
            )
        input_size = data["t_avg"].shape
        out = pd.DataFrame(
            {
                "time": np.tile(data["time"], input_size[1]),
                "id": np.tile(
                    np.arange(input_size[1]), (input_size[0], 1)
                ).T.flatten(),
                "t_avg": data["t_avg"].T.flatten(),
                "t_surf": data["t_surf"].T.flatten(),
                "t_core": data["t_core"].T.flatten(),
            }
        )
        return out


class ThermalSteadyResults(ThermalResults):
    """Thermal steady-state results parser."""

    def __init__(self, input_data: dict | pd.DataFrame):
        """Initialize steady-state thermal results.

        Args:
            input_data (dict | pd.DataFrame): Raw steady-state thermal results data.
        """
        super().__init__(input_data)

    @staticmethod
    def parse_results(data: dict | pd.DataFrame) -> pd.DataFrame:
        """Parse steady-state thermal results into a DataFrame.

        Converts raw steady-state thermal output into standardized DataFrame format.
        If input is already a DataFrame, returns it as-is. Otherwise converts dict to DataFrame.

        Args:
            data: Raw steady-state results as dictionary or DataFrame.

        Returns:
            Parsed results as a pandas DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data)


class ThermalForecastArray:
    """Array for input thermal forecast parameters."""

    # thl is strange to handle time series input TODO ?
    time = np.arange(10)
    wind_speed = np.linspace(0, 5, 10)
    ambient_temp = np.linspace(15, 25, 10)
    solar_irradiance = np.linspace(0, 800, 10)


# TODO: the temperature outputs have some parameters, perhpas properties are not the best way to handle that
# TODO: add latitude/longitude/altitude/azimuth in the section array
# TODO: add weather in the weather array ?
# TODO: warning, the thermal engine is using default parameters from thl, need to mirror that in mechaphlowers / future array structure ?
# TODO: conf array for intensity / target temperature ?
# TODO: builders for ThermalEngine from array
# TODO: add unit for ThermalEngine
# TODO: verify reactivity
# TODO: plot part


def normalize_inputs(
    **kwargs: float | np.ndarray | None,
) -> tuple[dict[str, np.ndarray | float], int | None]:
    """Normalize and validate input parameters.

    Converts scalar floats to arrays and ensures all array inputs have the same shape.
    Sets the __len__ attribute to the length of input vectors.

    Args:
        **kwargs: Input parameters as floats or arrays.

    Returns:
        dict: Dictionary with inputs as numpy arrays or floats.

    Raises:
        ValueError: If array inputs have incompatible shapes.
    """
    normalized: dict[str, np.ndarray | float] = {}
    array_length: int | None = None

    for key, value in kwargs.items():
        if value is None:
            normalized[key] = np.nan
        elif isinstance(value, (int, float)):
            # Convert scalar to array, will broadcast later if needed
            normalized[key] = np.asarray(value)
        elif isinstance(value, np.ndarray):
            normalized[key] = value
            # Track the length of array inputs
            if value.size > 1:
                if array_length is None:
                    array_length = value.size
                elif value.size != array_length:
                    raise ValueError(
                        f"All array inputs must have the same length. "
                        f"Expected {array_length}, got {value.size} for {key}."
                    )
        else:
            normalized[key] = np.asarray(value)

    return normalized, array_length


class ThermalEngine:
    """Thermal engine is a wrapper for cable thermal modeling."""

    available_power_model = {
        "rte": solver.rte,
    }
    available_heat_equation = {"3t": "3t"}

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
        self.heateq = self.available_heat_equation.get("3t", ValueError)
        self.dict_input = {}
        self.forecast = ThermalForecastArray()
        self.target_temperature = 65

    def set(
        self,
        cable_array: CableArray,
        latitude: float | np.ndarray,
        longitude: float | np.ndarray,
        altitude: float | np.ndarray,
        azimuth: float | np.ndarray,
        month: int | np.ndarray,
        day: int | np.ndarray,
        hour: int | np.ndarray,
        intensity: float | np.ndarray,
        ambient_temp: float | np.ndarray,
        wind_speed: float | np.ndarray,
        wind_angle: float | np.ndarray,
        solar_irradiance: float | np.ndarray | None = None,
    ):
        """Set input parameters for thermal calculations.

        Args:
            cable_array: An instance of CableArray containing cable properties.
            latitude: Latitude values.
            longitude: Longitude values.
            altitude: Altitude values.
            azimuth: Azimuth values.
            month: Month values.
            day: Day values.
            hour: Hour values.
            intensity: Current intensity values.
            ambient_temp: Ambient temperature values.
            wind_speed: Wind speed values.
            wind_angle: Wind angle values.
            solar_irradiance: Solar irradiance values (optional). Defaults to None.
        """
        # Normalize and validate all input parameters
        inputs, array_length = normalize_inputs(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            azimuth=azimuth,
            month=month,
            day=day,
            hour=hour,
            intensity=intensity,
            ambient_temp=ambient_temp,
            wind_speed=wind_speed,
            wind_angle=wind_angle,
            solar_irradiance=solar_irradiance,
        )

        # Set __len__ attribute if any array input was found
        self._len = array_length if array_length is not None else 1

        self.dict_input = {
            "Qs": inputs["solar_irradiance"],
            "lat": inputs["latitude"],
            "lon": inputs["longitude"],
            "alt": inputs["altitude"],
            "azm": inputs["azimuth"],
            "month": inputs["month"],
            "day": inputs["day"],
            "hour": inputs["hour"],
            "Ta": inputs["ambient_temp"],
            "ws": inputs["wind_speed"],  # wind speed (m.s**-1)
            "wa": inputs["wind_angle"],  # wind angle (deg, regarding north)
            "I": inputs["intensity"],
            "m": cable_array.data.linear_mass.iloc[0],
            "d": cable_array.data.diameter_heart.iloc[0],
            "D": cable_array.data.diameter.iloc[0],
            "a": cable_array.data.section_heart.iloc[0],
            "A": cable_array.data.section_conductor.iloc[0],
            "l": cable_array.data.radial_thermal_conductivity.iloc[0],
            "alpha": cable_array.data.solar_absorption.iloc[0],
            "epsilon": cable_array.data.emissivity.iloc[0],
            "RDC20": cable_array.data.electric_resistance_20.iloc[0],
            "kl": cable_array.data.linear_resistance_temperature_coef.iloc[0],
            "km": 1.006
            if cable_array.data.has_magnetic_heart.iloc[0]
            else 1.0,
            "ki": 0.016
            if cable_array.data.has_magnetic_heart.iloc[0]
            else 0.0,
        }
        self._load()

    def load(self):
        """Load or reload the thermal model, and checks the shape of the input parameters.
        Can be used if the input parameters are modified without using set()."""
        normalize_inputs(**self.dict_input)
        self._load()

    def _load(self):
        """Load the thermal model with the current input parameters."""
        # expected to fail if arguments are not filled
        self.thermal_model = self.power_model(
            dic=self.dict_input, heateq=self.heateq
        )

    def steady_temperature(
        self, intensity: np.ndarray | None = None
    ) -> ThermalSteadyResults:
        """Compute steady-state temperature results.

        Returns:
            ThermalSteadyResults: An instance containing steady-state temperature data.
        """
        if intensity is not None:
            self.dict_input["I"] = intensity
            self.load()
        return ThermalSteadyResults(self.thermal_model.steady_temperature())

    def steady_intensity(
        self, target_temperature: np.ndarray | None = None
    ) -> ThermalSteadyResults:
        """Compute steady-state intensity results.

        Returns:
            ThermalSteadyResults: An instance containing steady-state intensity data.
        """
        if target_temperature is not None:
            self.target_temperature = target_temperature

        return ThermalSteadyResults(
            self.thermal_model.steady_intensity(self.target_temperature)
        )

    def transient_temperature(
        self, forecast_control: ThermalForecastArray | None = None
    ) -> ThermalTransientResults:
        """Compute transient temperature results.

        Returns:
            ThermalTransientResults: An instance containing time-varying temperature data.
        """
        if forecast_control is not None:
            self.forecast = forecast_control

        return ThermalTransientResults(
            self.thermal_model.transient_temperature(time=self.forecast.time)
        )

    @property
    def wind_cable_angle(self) -> float | np.ndarray:
        """Compute the angle between wind and cable direction.

        Triggers ambient_wind_speed mode in models.

        Returns:
            Angle in degrees between wind direction and cable azimuth.
        """
        # TODO: move this into thl (formulae in thl.power.convective_cooling line 35)
        return np.rad2deg(
            np.arcsin(
                np.sin(
                    np.deg2rad(
                        np.abs(self.dict_input["azm"] - self.dict_input["wa"])
                        % 180.0
                    )
                )
            )
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
