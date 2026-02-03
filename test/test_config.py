# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]

from mechaphlowers.config import options
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.plotting.plot import PlotEngine


def test_config_on_plot(balance_engine_base_test) -> None:
    balance_engine_base_test.solve_adjustment()
    plt_line = PlotEngine(balance_engine_base_test)

    fig = go.Figure()
    original_res = options.graphics.resolution
    original_trace_profile = options.graphics.cable_trace_profile.copy()
    options.graphics.resolution = 20
    options.graphics.cable_trace_profile["size"] = 10.0
    plt_line.preview_line3d(fig)
    assert (
        fig._data[0].get('marker').get('size') == options.graphics.marker_size  # type: ignore[attr-defined]
    )
    assert fig._data[0].get('x').shape[0] == (  # type: ignore[attr-defined]
        options.graphics.resolution + 1
    ) * (balance_engine_base_test.section_array.data.shape[0] - 1)
    # fig.show() # deactivate for auto unit testing
    # restore original settings
    options.graphics.resolution = original_res
    options.graphics.cable_trace_profile = original_trace_profile


def test_change_values_input_unit__one_value() -> None:
    original_diameter_unit = options.input_units.cable_array["diameter"]
    original_angle_unit = options.input_units.section_array["line_angle"]

    options.input_units.cable_array["diameter"] = "cm"
    options.input_units.section_array["line_angle"] = "rad"

    expected_dict_cable = {
        "section": "mm^2",
        "diameter": "cm",
        "young_modulus": "MPa",
        "linear_mass": "kg/m",
        "dilatation_coefficient": "1/K",
        "temperature_reference": "°C",
        "a0": "MPa",
        "a1": "MPa",
        "a2": "MPa",
        "a3": "MPa",
        "a4": "MPa",
        "b0": "MPa",
        "b1": "MPa",
        "b2": "MPa",
        "b3": "MPa",
        "b4": "MPa",
        "diameter_heart": "mm",
        "section_conductor": "mm^2",
        "section_heart": "mm^2",
        "electric_resistance_20": "ohm.m**-1",
        "linear_resistance_temperature_coef": "K**-1",
        "radial_thermal_conductivity": "W.m**-1.K**-1",
    }

    expected_dict_section = {
        "conductor_attachment_altitude": "m",
        "crossarm_length": "m",
        "line_angle": "rad",
        "insulator_length": "m",
        "span_length": "m",
        "insulator_mass": "kg",
    }

    assert options.input_units.cable_array == expected_dict_cable
    assert options.input_units.section_array == expected_dict_section

    options.input_units.cable_array["diameter"] = original_diameter_unit
    options.input_units.section_array["line_angle"] = original_angle_unit


def test_input_unit__arrays() -> None:
    original_diameter_unit = options.input_units.cable_array["diameter"]
    original_angle_unit = options.input_units.section_array["line_angle"]

    options.input_units.cable_array["diameter"] = "cm"
    options.input_units.section_array["line_angle"] = "rad"

    cable_array = CableArray(
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
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2"],
                "suspension": [False, False],
                "conductor_attachment_altitude": [30, 40],
                "crossarm_length": [0, 0],
                "line_angle": [0, 0],
                "insulator_length": [0, 0],
                "span_length": [480, np.nan],
                "insulator_mass": np.array([1000, 1000]),
            }
        )
    )

    assert cable_array.input_units["diameter"] == "cm"
    assert section_array.input_units["line_angle"] == "rad"
    options.input_units.cable_array["diameter"] = original_diameter_unit
    options.input_units.section_array["line_angle"] = original_angle_unit
