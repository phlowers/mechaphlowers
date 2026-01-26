# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from mechaphlowers import (
    BalanceEngine,
    CableArray,
    PapotoParameterMeasure,
    PlotEngine,
    SectionArray,
    param_calibration,
)
from mechaphlowers.data.catalog import sample_cable_catalog


@pytest.mark.release_test
def test_create_cable_array():
    CableArray(
        pd.DataFrame(
            {
                "section": [345.55],
                "diameter": [22.4],
                "linear_mass": [0.974],
                "young_modulus": [59000],
                "dilatation_coefficient": [23e-6],
                "temperature_reference": [15],
                "a0": [0],
                "a1": [59000],
                "a2": [0],
                "a3": [0],
                "a4": [0],
                "b0": [0],
                "b1": [0],
                "b2": [0],
                "b3": [0],
                "b4": [0],
                "diameter_heart": [0.0],
                "section_conductor": [345.55],
                "section_heart": [0.0],
                "solar_absorption": [0.9],
                "emissivity": [0.8],
                "electric_resistance_20": [0.0554],
                "linear_resistance_temperature_coef": [0.0036],
                "is_polynomial": [False],
                "radial_thermal_conductivity": [1.0],
                "has_magnetic_heart": [False],
            }
        )
    )


@pytest.mark.release_test
def test_run_balance_engine():
    cable_AM600 = sample_cable_catalog.get_as_object(["ASTER600"])

    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": [0, 10, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [1000, 500, 500, 1000],
                "load_mass": [500, 1000, 500, np.nan],
                "load_position": [0.2, 0.4, 0.6, np.nan],
            }
        )
    )
    section_array.add_units({"line_angle": "grad"})
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    balance_engine = BalanceEngine(
        cable_array=cable_AM600, section_array=section_array
    )

    balance_engine.solve_adjustment()
    balance_engine.solve_change_state(new_temperature=90, wind_pressure=200)


@pytest.mark.release_test
def test_run_balance_engine_plot():
    cable_AM600 = sample_cable_catalog.get_as_object(["ASTER600"])

    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": [0, 10, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [1000, 500, 500, 1000],
                "load_mass": [500, 1000, 500, np.nan],
                "load_position": [0.2, 0.4, 0.6, np.nan],
            }
        )
    )
    section_array.add_units({"line_angle": "grad"})
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    balance_engine = BalanceEngine(
        cable_array=cable_AM600, section_array=section_array
    )

    balance_engine.solve_adjustment()
    balance_engine.solve_change_state(new_temperature=90, wind_pressure=200)
    plt_engine = PlotEngine(balance_engine)

    # get coordinates for plotting in 3D
    plt_engine.get_points_for_plot()
    # 2D
    plt_engine.get_points_for_plot(project=False, frame_index=1)

    fig = go.Figure()
    plt_engine.preview_line3d(fig)


@pytest.mark.release_test
def test_papoto_measure():
    a = 498.565922913587
    HL = 0.0
    VL = 97.4327311161033
    HR = 162.614599621714
    VR = 88.6907631859419
    H1 = 5.1134354937127
    V1 = 98.4518011880176
    H2 = 19.6314054626454
    V2 = 97.6289296721015
    H3 = 97.1475339907774
    V3 = 87.9335010245142

    papoto = PapotoParameterMeasure()
    papoto(
        a=a,
        HL=HL,
        VL=VL,
        HR=HR,
        VR=VR,
        H1=H1,
        V1=V1,
        H2=H2,
        V2=V2,
        H3=H3,
        V3=V3,
    )


@pytest.mark.release_test
def test_parameter_calibration():
    cable_AM600 = sample_cable_catalog.get_as_object(["ASTER600"])

    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": [0, 10, 0, 0],
                "insulator_length": [0.001, 3, 3, 0.001],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [0, 0, 0, 0],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    param_calibration(2000, 60, section_array, cable_AM600, span_index=0)
