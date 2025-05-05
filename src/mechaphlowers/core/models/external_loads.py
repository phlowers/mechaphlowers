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
    def __init__(
        # allow floats?
        self,
        wind_angle_cable_degrees: np.ndarray,
        tower_height: np.float64,  # or np.ndarray?
        voltage: np.float64,  # np.ndarray?
        category_surface_roughness: Literal["0", "II", "III"] = "II",  # IIIa?
        gust_wind: np.ndarray | None = None,
        speed_average_wind_open_country: np.ndarray | None = None,
        work: bool = False,
    ):
        self.wind_angle_cable_degrees = wind_angle_cable_degrees
        self.tower_height = tower_height
        self.voltage = voltage
        self.category_surface_roughness = category_surface_roughness
        if speed_average_wind_open_country is None:
            if gust_wind is None:
                raise TypeError(
                    "gust_wind or speed_average_wind_open_country need to be not"
                )
            else:
                speed_average_wind_open_country = gust_wind / 1.54 / 3.6
        # if gust_wind and speed_average_wind_open_country are both given, speed_average_wind_open_country is used
        self.gust_wind = gust_wind
        self._speed_average_wind_open_country = speed_average_wind_open_country
        self.work = work

    @property
    def speed_average_wind_open_country(self):
        return np.round(self._speed_average_wind_open_country, 1)

    def get_pressure(self) -> np.ndarray:
        if self.voltage <= 90:
            h = self.tower_height * 3 / 4
        else:
            h = self.tower_height * 2 / 3

        roughness = {"0": 0.005, "II": 0.05, "III": 0.2}
        length = np.float64(roughness[self.category_surface_roughness])

        k: np.float64 = 0.19 * (length / 0.05) ** 0.07
        V = (
            self._speed_average_wind_open_country
            * k
            * np.log(h / length)
            * np.sin(self.wind_angle_cable_degrees / 180 * math.pi)
        )
        Iv: np.float64 = 1 / np.log(h / length)

        work_coefficient = 1.0
        if self.work is True:
            work_coefficient = 1.2

        wind_load_pa = (
            0.5 * 1.25 * V**2 * (1 + 7 * Iv) * 2 / 3 * work_coefficient
        )
        return wind_load_pa

    def get_pressure_rounded(self) -> np.ndarray:
        return np.round(self.get_pressure() / 10) * 10
