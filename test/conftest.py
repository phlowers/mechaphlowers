# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest

projet_dir: Path = Path(__file__).resolve().parents[1]
source_dir: Path = projet_dir / "src"
sys.path.insert(0, str(source_dir))

from mechaphlowers.api.frames import SectionDataFrame
from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
    WeatherArray,
)
from mechaphlowers.entities.data_container import (
    DataContainer,
    factory_data_container,
)


@pytest.fixture
def default_cable_array() -> CableArray:
    return CableArray(
        pd.DataFrame(
            {
                "section": [345.55],
                "diameter": [22.4],
                "linear_weight": [9.55494],
                "young_modulus": [59],
                "dilatation_coefficient": [23],
                "temperature_reference": [15],
                "a0": [0],
                "a1": [59],
                "a2": [0],
                "a3": [0],
                "a4": [0],
                "b0": [0],
                "b1": [0],
                "b2": [0],
                "b3": [0],
                "b4": [0],
            }
        )
    )


@pytest.fixture
def default_section_array_one_span() -> SectionArray:
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
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
    return section_array


@pytest.fixture
def default_section_array_three_spans() -> SectionArray:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["support 1", "2", "three", "support 4"]),
                "suspension": np.array([False, True, True, False]),
                "conductor_attachment_altitude": np.array([2.2, 5, -0.12, 0]),
                "crossarm_length": np.array([10, 12.1, 10, 10.1]),
                "line_angle": np.array([0, 360, 90.1, -90.2]),
                "insulator_length": np.array([0, 4, 3.2, 0]),
                "span_length": np.array([400, 500.2, 500.0, np.nan]),
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
    return section_array


@pytest.fixture
def factory_neutral_weather_array() -> Callable[[int], WeatherArray]:
    def _method_cable_array(nb_span=2):
        return WeatherArray(
            pd.DataFrame(
                {
                    "ice_thickness": [0.0] * nb_span,
                    "wind_pressure": [0.0] * nb_span,
                }
            )
        )

    return _method_cable_array


@pytest.fixture
def generic_weather_array_three_spans() -> WeatherArray:
    return WeatherArray(
        pd.DataFrame(
            {
                "ice_thickness": [1, 2.1, 0.0, np.nan],
                "wind_pressure": [240.12, 0.0, 12.0, np.nan],
            }
        )
    )


@pytest.fixture
def default_data_container_one_span() -> DataContainer:
    NB_SPAN = 2
    cable_array = CableArray(
        pd.DataFrame(
            {
                "section": [345.55],
                "diameter": [22.4],
                "linear_weight": [9.55494],
                "young_modulus": [59],
                "dilatation_coefficient": [23],
                "temperature_reference": [15],
                "a0": [0],
                "a1": [59],
                "a2": [0],
                "a3": [0],
                "a4": [0],
                "b0": [0],
                "b1": [0],
                "b2": [0],
                "b3": [0],
                "b4": [0],
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
                "sagging_parameter": [2000, 2000],
                "sagging_temperature": [15, 15],
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    weather_array = WeatherArray(
        pd.DataFrame(
            {
                "ice_thickness": [0.0, 0.0],
                "wind_pressure": np.zeros(NB_SPAN),
            }
        )
    )

    data_container = factory_data_container(
        section_array, cable_array, weather_array
    )
    return data_container


@pytest.fixture
def section_dataframe_with_cable_weather() -> SectionDataFrame:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["support 1", "2", "three", "support 4"]),
                "suspension": np.array([False, True, True, False]),
                "conductor_attachment_altitude": np.array([2.2, 5, -0.12, 0]),
                "crossarm_length": np.array([10, 12.1, 10, 10.1]),
                "line_angle": np.array([0, 360, 90.1, -90.2]),
                "insulator_length": np.array([0, 4, 3.2, 0]),
                "span_length": np.array([400, 500.2, 500.0, np.nan]),
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    frame = SectionDataFrame(section_array)

    cable_array = CableArray(
        pd.DataFrame(
            {
                "section": [345.55],
                "diameter": [22.4],
                "linear_weight": [9.55494],
                "young_modulus": [59],
                "dilatation_coefficient": [23],
                "temperature_reference": [15],
                "a0": [0],
                "a1": [59],
                "a2": [0],
                "a3": [0],
                "a4": [0],
                "b0": [0],
                "b1": [0],
                "b2": [0],
                "b3": [0],
                "b4": [0],
            }
        )
    )
    frame.add_cable(cable_array)

    weather = WeatherArray(
        pd.DataFrame(
            {
                "ice_thickness": [1, 2.1, 0.0, np.nan],
                "wind_pressure": [1840.12, 0.0, 12.0, np.nan],
            }
        )
    )
    frame.add_weather(weather)

    return frame
