# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from thermohl import solver    # type: ignore

from mechaphlowers.entities.arrays import CableArray


class ThermalResults:
    pass

class ThermalTransientResults(ThermalResults):
    pass

class ThermalSteadyResults(ThermalResults):
    pass

class ThermalForecastArray:
    time = np.arange(10)
    

class ThermalEngine:
    
    available_power_model = {"rte": solver.rte,}
    available_heat_equation = {"3t": "3t"}
    
    def __init__(self):
        self.power_model = self.available_power_model.get("rte", ValueError)
        self.heateq= self.available_heat_equation.get("3t", ValueError)
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
            "km": 1.006 if cable_array.data.has_magnetic_heart.iloc[0] else 1.0,
            "ki": 0.016 if cable_array.data.has_magnetic_heart.iloc[0] else 0.0,
        }
        
    def load(self):
        # expected to fail if arguments are not filled
        self.span = self.power_model(dic=self.dict_input, heateq=self.heateq)
        
    @property
    def steady_temperature(self):
        return self.span.steady_temperature()
    
    @property
    def steady_intensity(self):
        return self.span.steady_intensity(
            self.target_temperature
        )
    
    @property
    def transient_temperature(self):
        return self.span.transient_temperature(
            time = self.forecast.time
            )
    
        
    

    # def get_cable_temperature(self):
    #     slvr = self.power_model(dic=self.dict_input, heateq=self.heateq)
    #     temp = slvr.steady_temperature()
    #     return temp.t_avg.to_numpy()
    
    # span_th.span.transient_temperature(time=np.arange(10))
