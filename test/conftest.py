# Copyright (c) 2026, RTE (http://www.rte-france.com)
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

from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.data.catalog.catalog import (
    sample_cable_catalog,
)
from mechaphlowers.data.units import convert_weight_to_mass

projet_dir: Path = Path(__file__).resolve().parents[1]
source_dir: Path = projet_dir / "src"
sys.path.append(str(source_dir))

from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
    WeatherArray,
)
from mechaphlowers.entities.data_container import (
    DataContainer,
    factory_data_container,
)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Capture test outcome and update benchmark report."""
    outcome = yield
    rep = outcome.get_result()

    if rep.when == "call" and "benchmark_report" in item.funcargs:
        benchmark_report = item.funcargs["benchmark_report"]

        # Find the entry index that was just recorded for this test
        if hasattr(item, "_benchmark_entry_idx"):
            idx = item._benchmark_entry_idx
            benchmark_report["tests"][idx]["status"] = rep.outcome
            if rep.outcome == "failed" and rep.longrepr:
                benchmark_report["tests"][idx]["error"] = str(rep.longrepr)


@pytest.fixture
def default_cable_array() -> CableArray:
    return CableArray(
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
                "insulator_mass": np.array([1000, 1000]),
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
                "insulator_mass": np.array([1000.0, 500.0, 500.0, 1000.0]),
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
    section_array.add_units({"line_angle": "deg"})
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
    cable_array = CableArray(
        pd.DataFrame(
            {
                "section": [600.4],
                "diameter": [31.86],
                "linear_mass": [1.8],
                "young_modulus": [60000],
                "dilatation_coefficient": [23e-6],
                "temperature_reference": [15.0],
                "a0": [0.0],
                "a1": [60000],
                "a2": [0.0],
                "a3": [0.0],
                "a4": [0.0],
                "b0": [0.0],
                "b1": [0.0],
                "b2": [0.0],
                "b3": [0.0],
                "b4": [0.0],
                "diameter_heart": [0.0],
                "section_conductor": [600.4],
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
                "sagging_parameter": [2000, 2000],
                "sagging_temperature": [15, 15],
                "insulator_mass": np.array([1000, 1000]),
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
    section_array.add_units({"line_angle": "deg"})

    weather_array = WeatherArray(
        pd.DataFrame(
            {
                "ice_thickness": [0.0, 0.0],
                "wind_pressure": [0.0, 0.0],
            }
        )
    )

    data_container = factory_data_container(
        section_array, cable_array, weather_array
    )
    return data_container



@pytest.fixture
def cable_array_AM600() -> CableArray:
    return sample_cable_catalog.get_as_object(["ASTER600"])


@pytest.fixture
def cable_array_NARCISSE600G() -> CableArray:
    return sample_cable_catalog.get_as_object(["NARCISSE600G"])


@pytest.fixture
def balance_engine_base_test(cable_array_AM600: CableArray) -> BalanceEngine:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [50, 100, 50, 50],
                "crossarm_length": [10, 10, 10, 10],
                "line_angle": [0, 0, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 500, 500, np.nan],
                "insulator_mass": convert_weight_to_mass(
                    [1000.0, 500.0, 500.0, 1000.0]
                ),
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        )
    )
    section_array.add_units({"line_angle": "grad"})
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
    return BalanceEngine(
        cable_array=cable_array_AM600, section_array=section_array
    )


@pytest.fixture
def balance_engine_angles(cable_array_AM600: CableArray) -> BalanceEngine:
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [50, 100, 50, 60],
                "crossarm_length": [10, 10, 10, 10],
                "line_angle": [0, 10, 20, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 500, 500, np.nan],
                "insulator_mass": convert_weight_to_mass(
                    [1000.0, 500.0, 500.0, 1000.0]
                ),
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        )
    )
    section_array.add_units({"line_angle": "grad"})
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
    return BalanceEngine(
        cable_array=cable_array_AM600, section_array=section_array
    )
