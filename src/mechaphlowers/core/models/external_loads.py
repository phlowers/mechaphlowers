# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import math
from math import pi
from typing import Any, Literal

import numpy as np

DEFAULT_ICE_DENSITY = 6_000


class CableLoads:
    """CableLoads is a class that allows to calculate the loads on the cable due to wind and ice

    Args:
            diameter (np.float64): diameter of the cable
            linear_weight (np.float64): linear weight of the cable
            ice_thickness (np.ndarray): thickness of the ice on the cable
            wind_pressure (np.ndarray): wind pressure on the cable
            ice_density (float, optional): density of the ice. Defaults to DEFAULT_ICE_DENSITY.
            **kwargs (Any, optional): additional arguments


    """

    def __init__(
        self,
        diameter: np.float64,
        linear_weight: np.float64,
        ice_thickness: np.ndarray,
        wind_pressure: np.ndarray,
        ice_density: float = DEFAULT_ICE_DENSITY,
        **kwargs: Any,
    ) -> None:
        self.diameter = diameter
        self.linear_weight = linear_weight
        self.ice_thickness = ice_thickness
        self.wind_pressure = wind_pressure
        self.ice_density = ice_density

    @property
    def load_angle(self) -> np.ndarray:
        """Load angle (in radians)

        Returns:
                np.ndarray: load angle (beta) for each span
        """
        linear_weight = self.linear_weight
        ice_load = self.ice_load
        wind_load = self.wind_load

        return np.arctan(wind_load / (ice_load + linear_weight))

    @property
    def resulting_norm(
        self,
    ) -> np.ndarray:
        """Norm of the force (R) applied on the cable due to weather loads and cable own weight, per meter cable"""

        linear_weight = self.linear_weight
        ice_load = self.ice_load
        wind_load = self.wind_load

        return np.sqrt((ice_load + linear_weight) ** 2 + wind_load**2)

    @property
    def load_coefficient(self) -> np.ndarray:
        linear_weight = self.linear_weight
        return self.resulting_norm / linear_weight

    @property
    def ice_load(self) -> np.ndarray:
        """Linear weight of the ice on the cable

        Returns:
                np.ndarray: linear weight of the ice for each span
        """
        e = self.ice_thickness
        D = self.diameter
        return self.ice_density * pi * e * (e + D)

    @property
    def wind_load(self) -> np.ndarray:
        """Linear force applied on the cable by the wind.

        Returns:
                np.ndarray: linear force applied on the cable by the wind
        """
        P_w = self.wind_pressure
        D = self.diameter
        e = self.ice_thickness
        return P_w * (D + 2 * e)

    def update_from_dict(self, data: dict) -> None:
        """Update the attributes of the instance based on a dictionary.

        Args:
                data (dict): Dictionary containing attribute names as keys and their values.
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


class WindSpeedPressureConverter:
    """WindSpeedPressureConverter is a class that allows to convert wind speed to wind pressure

    Args:
        tower_height (np.ndarray): height of the tower in meters
        gust (np.ndarray | None, optional): gust wind speed in km/h. Defaults to None.
        speed_average_open_country (np.ndarray | None, optional): average wind speed in open country in m/s. Defaults to None.
        angle_cable_degrees (np.ndarray, optional): angle of the wind on the cable in degrees. Defaults to 90.
        voltage (int, optional): voltage of the line in kV. Defaults to 400.
        category_surface_roughness (Literal["0", "II", "IIIa"], optional): category of surface roughness. Defaults to "II".
        work (bool, optional): if True, the converter is used for work conditions. Defaults to False.
    """

    def __init__(
        self,
        tower_height: np.ndarray,  # in m
        gust: np.ndarray | None = None,  # in km/h
        speed_average_open_country: np.ndarray | None = None,  # in m/s
        angle_cable_degrees: np.ndarray | None = None,
        voltage: int = 400,  # in kV
        category_surface_roughness: Literal["0", "II", "IIIa"] = "II",
        work: bool = False,
    ):
        self.tower_height = tower_height
        self.tower_height_max = np.max(tower_height)
        if speed_average_open_country is None:
            if gust is None:
                raise TypeError(
                    "gust_wind or speed_average_wind_open_country need to be not None"
                )
            else:
                speed_average_open_country = gust / 1.54 / 3.6
        # if gust_wind and speed_average_wind_open_country are both given, speed_average_wind_open_country is used
        self.gust = gust
        self._speed_average_open_country = speed_average_open_country
        if angle_cable_degrees is None:
            angle_cable_degrees = np.full_like(90, speed_average_open_country)
        self.angle_cable_degrees = angle_cable_degrees
        self.voltage = voltage
        self.category_surface_roughness = category_surface_roughness
        self.work = work

    @property
    def speed_average(self) -> np.ndarray:
        """Returns a rounded value of the average wind speed in open country to the nearest tenth. Value in m/s
        This value is used for display purposes, but the actual value used in calculations is not rounded.
        """
        return np.round(self._speed_average_open_country, 1)

    @property
    def pressure(self) -> np.ndarray:
        """Calculates the wind pressure in Pa based on the average wind speed, max tower height, voltage, surface roughness, and work condition."""
        if self.voltage <= 90:
            h = self.tower_height_max * 3 / 4
        else:
            h = self.tower_height_max * 2 / 3

        roughness = {"0": 0.005, "II": 0.05, "IIIa": 0.2}
        # roughness distance
        z0 = np.float64(roughness[self.category_surface_roughness])
        # terrain factor
        k_r: np.float64 = 0.19 * (z0 / 0.05) ** 0.07
        V_m = (
            self._speed_average_open_country
            * k_r
            * np.log(h / z0)
            * np.sin(self.angle_cable_degrees / 180 * math.pi)
        )
        # Iv: turbulence intensity
        Iv: np.float64 = 1 / np.log(h / z0)
        force_coefficient = 1.0
        if self.work is True:
            force_coefficient = 1.2

        # air density
        rho = 1.25

        wind_load_pa = (
            0.5 * rho * V_m**2 * (1 + 7 * Iv) * 2 / 3 * force_coefficient
        )
        return wind_load_pa

    @property
    def pressure_rounded(self) -> np.ndarray:
        """Returns the wind pressure rounded to the nearest 10 Pa."""
        return np.round(self.pressure / 10) * 10
