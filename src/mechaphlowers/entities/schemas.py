# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Optional

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import pandas as pdt


class SectionArrayInput(pa.DataFrameModel):
    """Schema for the data expected for a dataframe used to instantiate a SectionArray.

    Each row describes a support and the following span (except the last row which "only" describes the last support).

    Notes:
        Line angles are expressed in degrees.

        insulator_length should be zero for the first and last supports, since for now mechaphlowers
        ignores them when computing the state of a span or section.
        Taking them into account might be implemented later.

        span_length should be zero or numpy.nan for the last row.
    """

    name: pdt.Series[str]
    suspension: pdt.Series[bool]
    conductor_attachment_altitude: pdt.Series[float] = pa.Field(coerce=True)
    crossarm_length: pdt.Series[float] = pa.Field(coerce=True)
    line_angle: pdt.Series[float] = pa.Field(coerce=True)
    insulator_length: pdt.Series[float] = pa.Field(coerce=True)
    span_length: pdt.Series[float] = pa.Field(nullable=True, coerce=True)
    insulator_mass: pdt.Series[float] = pa.Field(coerce=True)
    load_mass: Optional[pdt.Series[float]] = pa.Field(
        nullable=True,
        coerce=True,
    )
    load_position: Optional[pdt.Series[float]] = pa.Field(
        nullable=True, coerce=True
    )
    ground_altitude: Optional[pdt.Series[float]] = pa.Field(
        nullable=True, coerce=True
    )

    @pa.dataframe_check(
        description="""Each row in the dataframe contains information about a support
        and the span next to it, except the last support which doesn't have a "next" span.
        So, specifying a span_length in the last row doesn't make any sense.
        Please set span_length to "not a number" (numpy.nan) to suppress this error.""",
    )
    def no_span_length_for_last_row(cls, df: pd.DataFrame) -> bool:
        return df.tail(1)["span_length"].isin([0, np.nan]).all()


class CableArrayInput(pa.DataFrameModel):
    """Schema for the data expected for a dataframe used to instantiate a CableArray.

    Attributes:
            section (float): Area of the section, in mm²
            diameter (float): Diameter of the cable, in mm
            linear_weight (float): Linear weight, in N/m
            young_modulus (float): Young modulus in GPa
            dilatation_coefficient (float): Dilatation coefficient in 10⁻⁶/°C
            temperature_reference (float): Temperature used to compute unstressed cable length (usually 0°C or 15°C)
            a0/a1/a2/a3/a4 (float): Coefficients of the relation between stress $\\sigma$ and deformation $\\varepsilon$ for the conductor: $\\sigma = a0 + a1*\\varepsilon + a2*\\varepsilon^2 + a3*\\varepsilon^3 + a4*\\varepsilon^4$
            b0/b1/b2/b3/b4 (float): Coefficients of the relation between stress $\\sigma$ and deformation $\\varepsilon$ for the heart: $\\sigma = b0 + b1*\\varepsilon + b2*\\varepsilon^2 + b3*\\varepsilon^3 + b4*\\varepsilon^4$
    """

    section: pdt.Series[float] = pa.Field(coerce=True)
    diameter: pdt.Series[float] = pa.Field(coerce=True)
    linear_mass: pdt.Series[float] = pa.Field(coerce=True)
    young_modulus: pdt.Series[float] = pa.Field(coerce=True)
    dilatation_coefficient: pdt.Series[float] = pa.Field(coerce=True)
    temperature_reference: pdt.Series[float] = pa.Field(coerce=True)
    a0: pdt.Series[float] = pa.Field(coerce=True)
    a1: pdt.Series[float] = pa.Field(coerce=True)
    a2: pdt.Series[float] = pa.Field(coerce=True)
    a3: pdt.Series[float] = pa.Field(coerce=True)
    a4: pdt.Series[float] = pa.Field(coerce=True)
    b0: pdt.Series[float] = pa.Field(coerce=True)
    b1: pdt.Series[float] = pa.Field(coerce=True)
    b2: pdt.Series[float] = pa.Field(coerce=True)
    b3: pdt.Series[float] = pa.Field(coerce=True)
    b4: pdt.Series[float] = pa.Field(coerce=True)
    diameter_heart: pdt.Series[float] = pa.Field(coerce=True)
    section_heart: pdt.Series[float] = pa.Field(coerce=True)
    section_conductor: pdt.Series[float] = pa.Field(coerce=True)
    solar_absorption: pdt.Series[float] = pa.Field(coerce=True)
    emissivity: pdt.Series[float] = pa.Field(coerce=True)
    electric_resistance_20: pdt.Series[float] = pa.Field(coerce=True)
    linear_resistance_temperature_coef: pdt.Series[float] = pa.Field(
        coerce=True
    )
    is_polynomial: pdt.Series[bool]
    radial_thermal_conductivity: pdt.Series[float] = pa.Field(coerce=True)
    has_magnetic_heart: pdt.Series[bool]


class WeatherArrayInput(pa.DataFrameModel):
    """Schema describing the expected dataframe for instantiating a WeatherArray.

    Attributes:
            ice_thickness (float): Thickness of the ice layer on the cable, in cm
            wind_pressure (float): Pressure of the perpendicular component of the wind, in Pa
    """

    ice_thickness: pdt.Series[float] = pa.Field(
        coerce=True, ge=0.0, nullable=True
    )
    wind_pressure: pdt.Series[float] = pa.Field(coerce=True, nullable=True)
