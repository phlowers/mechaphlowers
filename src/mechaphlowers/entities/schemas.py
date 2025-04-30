# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from typing import Optional

import numpy as np
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

    @pa.dataframe_check(
        description="""Though tension supports also have insulators,
        for now we ignore them when computing the state of a span or section.
        Taking them into account might be implemented later.
        For now, set the insulator length to 0 for tension supports to suppress this error."""
    )
    def insulator_length_is_zero_if_not_suspension(
        cls, df: pdt.DataFrame
    ) -> pdt.Series[bool]:
        return (df["suspension"] | (df["insulator_length"] == 0)).pipe(
            pdt.Series[bool]
        )

    @pa.dataframe_check(
        description="""Each row in the dataframe contains information about a support
        and the span next to it, except the last support which doesn't have a "next" span.
        So, specifying a span_length in the last row doesn't make any sense.
        Please set span_length to "not a number" (numpy.nan) to suppress this error.""",
    )
    def no_span_length_for_last_row(cls, df: pdt.DataFrame) -> bool:
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
    linear_weight: pdt.Series[float] = pa.Field(coerce=True)
    young_modulus: pdt.Series[float] = pa.Field(coerce=True)
    young_modulus_conductor: Optional[pdt.Series[float]] = pa.Field(
        coerce=True
    )
    young_modulus_heart: Optional[pdt.Series[float]] = pa.Field(coerce=True)
    dilatation_coefficient: pdt.Series[float] = pa.Field(coerce=True)
    dilatation_coefficient_conductor: Optional[pdt.Series[float]] = pa.Field(
        coerce=True
    )
    dilatation_coefficient_heart: Optional[pdt.Series[float]] = pa.Field(
        coerce=True
    )
    temperature_reference: pdt.Series[float] = pa.Field(coerce=True)
    section_conductor: Optional[pdt.Series[float]] = pa.Field(coerce=True)
    a0: Optional[pdt.Series[float]] = pa.Field(coerce=True)
    a1: Optional[pdt.Series[float]] = pa.Field(coerce=True)
    a2: Optional[pdt.Series[float]] = pa.Field(coerce=True)
    a3: Optional[pdt.Series[float]] = pa.Field(coerce=True)
    a4: Optional[pdt.Series[float]] = pa.Field(coerce=True)
    b0: Optional[pdt.Series[float]] = pa.Field(coerce=True)
    b1: Optional[pdt.Series[float]] = pa.Field(coerce=True)
    b2: Optional[pdt.Series[float]] = pa.Field(coerce=True)
    b3: Optional[pdt.Series[float]] = pa.Field(coerce=True)
    b4: Optional[pdt.Series[float]] = pa.Field(coerce=True)


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
