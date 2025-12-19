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

from mechaphlowers.entities.arrays import CableArray

logger = logging.getLogger(__name__)


class ThermalResults(ABC):
    """Thermal results base class."""

    def __init__(self, input_data: dict | pd.DataFrame):
        self.data = self.parse_results(input_data)

    @staticmethod
    @abstractmethod
    def parse_results(data: dict | pd.DataFrame) -> pd.DataFrame:
        pass


class ThermalTransientResults(ThermalResults):
    def __init__(self, input_data):
        super().__init__(input_data)

    @staticmethod
    def parse_results(data: dict | pd.DataFrame) -> pd.DataFrame:
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
    def __init__(self, input_data):
        super().__init__(input_data)

    @staticmethod
    def parse_results(data: dict | pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame(data)


class ThermalForecastArray:
    # thl is strange to handle time series input TODO ?
    time = np.arange(10)
    wind_speed = np.linspace(0, 5, 10)
    ambient_temp = np.linspace(15, 25, 10)
    solar_irradiance = np.linspace(0, 800, 10)


# TODO: the temperature outputs have some parameters, perhpas properties are not the best way to handle that
# TODO: add latitude/longitude/altitude/azimuth in the section array
# TODO: add weather in the weather array
# TODO: conf array for intensity / target temperature ?
# TODO: builders for ThermalEngine from array
# TODO: add unit for ThermalEngine
# TODO: add docstrings
# TODO: verify reactivity
# TODO: plot part


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
            target_temperature: Target temperature for steady-state calculations.
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
        if solar_irradiance is None:
            solar_irradiance = np.nan
        self.dict_input = {
            "Qs": solar_irradiance,
            "lat": latitude,
            "lon": longitude,
            "alt": altitude,
            "azm": azimuth,
            "month": month,
            "day": day,
            "hour": hour,
            "Ta": ambient_temp,
            "ws": wind_speed,  # wind speed (m.s**-1)
            "wa": wind_angle,  # wind angle (deg, regarding north)
            "I": intensity,
            "m": cable_array.data.linear_weight.iloc[0],
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
        self.load()

    def load(self):
        """Load or reload the thermal model with the current input parameters."""
        # expected to fail if arguments are not filled
        self.span = self.power_model(dic=self.dict_input, heateq=self.heateq)

    def steady_temperature(self) -> ThermalSteadyResults:
        """Compute steady-state temperature results."""
        return ThermalSteadyResults(self.span.steady_temperature())

    def steady_intensity(self) -> ThermalSteadyResults:
        """Compute steady-state intensity results."""
        return ThermalSteadyResults(
            self.span.steady_intensity(self.target_temperature)
        )

    def transient_temperature(self) -> ThermalTransientResults:
        """Compute transient temperature results."""
        return ThermalTransientResults(
            self.span.transient_temperature(time=self.forecast.time)
        )

    @property
    def wind_cable_angle(self) -> float | np.ndarray:
        """property triggering ambient_wind_speed mode in models"""
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
        """property triggering normal_wind mode in models. Not implemented yet."""
        raise NotImplementedError

    @normal_wind_mode.setter
    def normal_wind_mode(self, value: bool):
        """normal_wind_mode

        property triggering normal_wind mode in models. Not implemented yet.

        Args:
            value (bool): calculus is in normal_wind mode
        """
        # TODO: same than no wind mode but only for angle
        try:
            if not isinstance(value, bool):
                raise TypeError
            self._normal_wind_mode = bool(value)
        except TypeError:
            logger.warning("normal_wind_mode is expected boolean")
