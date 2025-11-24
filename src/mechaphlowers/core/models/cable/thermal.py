# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from thermohl import solver  # type: ignore

from mechaphlowers.entities.arrays import CableArray


def get_cable_temperature(cable_array: CableArray, month:int, day:int, hour:int, ambient_temp: float, wind_speed: float, wind_angle: float, heateq='3t'):
    dict_input = {
        "month": month,
        "day": day,
        "hour": hour,
        "Ta": ambient_temp,
        "ws" : wind_speed,  # wind speed (m.s**-1)
        "wa" : wind_angle,  # wind angle (deg, regarding north)
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
    slvr = solver.rte(dic=dict_input, heateq=heateq)
    temp = slvr.steady_temperature()
    return temp.t_avg.to_numpy()
