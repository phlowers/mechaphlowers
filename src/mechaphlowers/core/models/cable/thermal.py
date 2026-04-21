# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from math import floor
from typing import Any

import numpy as np
import pandas as pd
from pydantic import validate_call
from thermohl import solver  # type: ignore
from thermohl.solver.entities import (  # type: ignore
    PowerType,
    TemperatureType,
    VariableType,
)
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
        input_size = data[TemperatureType.AVERAGE].shape
        return pd.DataFrame(
            {
                "time": np.tile(data[VariableType.TIME], input_size[1]),
                "id": np.tile(
                    np.arange(input_size[1]), (input_size[0], 1)
                ).T.flatten(),
                "t_avg": data[TemperatureType.AVERAGE].T.flatten(),
                "t_surf": data[TemperatureType.SURFACE].T.flatten(),
                "t_core": data[TemperatureType.CORE].T.flatten(),
            }
        )


class ThermalSteadyResults(ThermalResults):
    """Thermal steady-state results parser."""

    def __init__(self, input_data: dict | pd.DataFrame):
        """Initialize steady-state thermal results.

        Args:
            input_data (dict | pd.DataFrame): Raw steady-state thermal results data.
        """
        super().__init__(input_data)

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
        column_names = {
            VariableType.TRANSIT: "transit",
            TemperatureType.SURFACE: "t_surf",
            TemperatureType.AVERAGE: "t_avg",
            TemperatureType.CORE: "t_core",
            PowerType.JOULE: "P_joule",
            PowerType.SOLAR: "P_solar",
            PowerType.CONVECTION: "P_convection",
            PowerType.RADIATION: "P_radiation",
            PowerType.RAIN: "P_precipitation",
        }
        if isinstance(data, pd.DataFrame):
            return data.rename(columns=column_names)
        return pd.DataFrame(
            {column_names.get(key, key): value for key, value in data.items()}
        )


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
    month: np.ndarray[Any, np.dtype[np.integer]] | None = None,
    day: np.ndarray[Any, np.dtype[np.integer]] | None = None,
    hour: np.ndarray[Any, np.dtype[np.integer] | np.dtype[np.floating]]
    | None = None,
    datetime_utc: list[datetime] | None = None,
    **kwargs: np.ndarray[Any, Any] | list[datetime],
) -> tuple[dict[str, np.ndarray[Any, Any] | list[datetime]], int]:
    """Validate input parameters.

    Ensures all inputs are numpy arrays with the same size and that
    datetime-related inputs have the right type. Also ensures that
    month, day and hour, if given, are in the right range.

    Args:
        **kwargs: Input parameters as numpy arrays.

    Returns:
        tuple: A tuple containing:
            - dict: Dictionary with the input numpy arrays.
            - int: The common length of all arrays.

    Raises:
        ValueError: If array inputs have incompatible sizes.
        TypeError: If any input is not a numpy array.
    """

    array_length: int | None = None

    for key, value in kwargs.items():
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Expected numpy array for '{key}', got {type(value).__name__}."
            )

        # Track and validate the length of array inputs
        if array_length is None:
            array_length = value.size
        elif value.size != array_length:
            raise ValueError(
                f"All array inputs must have the same length. "
                f"Expected {array_length}, got {value.size} for {key}."
            )

    array_length = check_datetime_arguments(
        array_length,
        month=month,
        day=day,
        hour=hour,
        datetime_utc=datetime_utc,
    )

    if month is not None:
        kwargs["month"] = month
    if day is not None:
        kwargs["day"] = day
    if hour is not None:
        kwargs["hour"] = hour
    if datetime_utc is not None:
        kwargs["datetime_utc"] = datetime_utc

    return kwargs, array_length


@validate_call
def check_datetime_arguments(
    array_length: int | None,
    month: np.ndarray[Any, np.dtype[np.integer]] | None,
    day: np.ndarray[Any, np.dtype[np.integer]] | None,
    hour: np.ndarray[Any, np.dtype[np.integer | np.floating]] | None,
    datetime_utc: list[datetime] | None,
) -> int:
    if datetime_utc is not None and (
        month is not None or day is not None or hour is not None
    ):
        raise ValueError(
            "Cannot provide both 'datetime_utc' and individual 'month', 'day', or 'hour' arrays."
        )

    for arg_name, value in [("month", month), ("day", day), ("hour", hour)]:
        if value is not None:
            if array_length is None:
                array_length = value.size
            elif value.size != array_length:
                raise ValueError(
                    f"All array inputs must have the same length. "
                    f"Expected {array_length}, got {value.size} for {arg_name}."
                )
    if datetime_utc is not None:
        if array_length is None:
            array_length = len(datetime_utc)
        elif len(datetime_utc) != array_length:
            raise ValueError(
                f"All array inputs must have the same length. "
                f"Expected {array_length}, got {len(datetime_utc)} for 'datetime_utc'."
            )

    check_datetime_argument_ranges(
        month=month,
        day=day,
        hour=hour,
    )

    if array_length is None:
        array_length = 0

    return array_length


def check_datetime_argument_ranges(
    month: np.ndarray[Any, np.dtype[np.integer]] | None,
    day: np.ndarray[Any, np.dtype[np.integer]] | None,
    hour: np.ndarray[Any, np.dtype[np.integer | np.floating]] | None,
) -> None:
    if month is not None:
        if not np.all((1 <= month) & (month <= 12)):
            raise ValueError(
                "Month values must be in the range 1-12. Invalid values found in 'month'."
            )
    if day is not None:
        if not np.all((1 <= day) & (day <= 31)):
            raise ValueError(
                "Day values must be in the range 1-31. Invalid values found in 'day'."
            )
    if hour is not None:
        if not np.all((0 <= hour) & (hour < 24)):
            raise ValueError(
                "Hour values must be in the range [0-24[. Invalid values found in 'hour'."
            )


def to_datetime(
    month: np.integer,
    day: np.integer,
    hour: np.floating,
) -> datetime:
    """Convert month, day, hour to a datetime object.

    Args:
        month (np.integer): Month value ([1-12]).
        day (np.integer): Day value ([1-31]).
        hour (np.floating): Hour value ([0-24[). Can include non-integer hours (e.g. 14.5 for 2:30 PM).

    Returns:
        datetime: A datetime object representing the given month, day, and hour.
    """
    return datetime(
        1970,  # Arbitrary year since it's not used in calculations
        month,
        day,
        hour=floor(hour),
        minute=floor((hour % 1) * 60),
        second=floor(((hour % 1) * 60) % 1 * 60),
        microsecond=floor((((hour % 1) * 60) % 1 * 60) % 1 * 1000000),
    )


class ThermalEngine:
    """Thermal engine is a wrapper for cable thermal modeling."""

    available_power_model = {
        "rte": solver.rte,
    }
    available_heat_equation = {
        "3t": solver.HeatEquationType.THREE_TEMPERATURES
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
        self.heateq = self.available_heat_equation.get("3t", ValueError)
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
        month: np.ndarray,
        day: np.ndarray,
        hour: np.ndarray,
        intensity: np.ndarray,
        ambient_temp: np.ndarray,
        wind_speed: np.ndarray,
        wind_angle: np.ndarray,
        solar_irradiance: np.ndarray | None = None,
    ):
        """Set input parameters for thermal calculations.

        Args:
            cable_array (CableArray): An instance of CableArray containing cable properties.
            latitude (np.ndarray): Latitude values.
            longitude (np.ndarray): Longitude values.
            altitude (np.ndarray): Altitude values.
            azimuth (np.ndarray): Azimuth values.
            month (np.ndarray): Month values.
            day (np.ndarray): Day values.
            hour (np.ndarray): Hour values.
            intensity (np.ndarray): Current intensity values.
            ambient_temp (np.ndarray): Ambient temperature values.
            wind_speed (np.ndarray): Wind speed values.
            wind_angle (np.ndarray): Wind angle values.
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
            month=month,
            day=day,
            hour=hour,
            intensity=intensity,
            ambient_temp=ambient_temp,
            wind_speed=wind_speed,
            wind_angle=wind_angle,
            solar_irradiance=solar_irradiance,
        )

        if inputs.get("datetime_utc") is None:
            datetime_utc = [
                to_datetime(month, day, hour)
                for month, day, hour in zip(
                    inputs["month"], inputs["day"], inputs["hour"]
                )
            ]
        else:
            datetime_utc = inputs.get("datetime_utc")  # type: ignore

        self.dict_input = {
            "measured_global_radiation": inputs["solar_irradiance"],
            "latitude": inputs["latitude"],
            "longitude": inputs["longitude"],
            "altitude": inputs["altitude"],
            "cable_azimuth": inputs["azimuth"],
            "datetime_utc": datetime_utc,
            "ambient_temperature": inputs["ambient_temp"],
            "wind_speed": inputs["wind_speed"],  # wind speed (m.s**-1)
            "wind_azimuth": inputs[
                "wind_angle"
            ],  # wind angle (deg, regarding north)
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
            heat_equation=self.heateq,  # TODO
        )

    def steady_temperature(
        self, intensity: np.ndarray | None = None
    ) -> ThermalSteadyResults:
        """Compute steady-state temperature results.

        Returns:
            ThermalSteadyResults: An instance containing steady-state temperature data.
        """
        logger.debug("Get steady_temperature()")
        if intensity is not None:
            self.dict_input["transit"] = intensity
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
            self.thermal_model.transient_temperature(offset=self.forecast.time)
        )

    @property
    def wind_cable_angle(self) -> float | np.ndarray:
        """Compute the angle between wind and cable direction.

        Triggers ambient_wind_speed mode in models.

        Returns:
            Angle in degrees between wind direction and cable azimuth.
        """
        # TODO: use wind_angle computed by thermohl?
        # TODO: move this into thl (formulae in thl.power.convective_cooling line 35)
        return np.rad2deg(
            np.arcsin(
                np.sin(
                    np.deg2rad(
                        np.abs(
                            self.dict_input["cable_azimuth"]
                            - self.dict_input["wind_azimuth"]
                        )
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
