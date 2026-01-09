# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd
import pytest

from mechaphlowers.core.models.cable.thermal import (
    ThermalEngine,
    ThermalTransientResults,
)
from mechaphlowers.entities.arrays import CableArray


@pytest.fixture
def thermal_engine_3_spans(cable_array_AM600: CableArray) -> ThermalEngine:
    thermal_engine = ThermalEngine()

    thermal_engine.set(
        cable_array_AM600,
        latitude=np.array([45.0, 45.0, 45.0]),
        longitude=np.array([0.0, 0.0, 0.0]),
        altitude=np.array([0.0, 0.0, 0.0]),
        azimuth=np.array([0.0, 0.0, 90.0]),
        month=np.array(
            [
                3,
                3,
                3,
            ]
        ),
        day=np.array(
            [
                21,
                21,
                21,
            ]
        ),
        hour=np.array(
            [
                22,
                22,
                12,
            ]
        ),
        intensity=np.array([100.0, 1000.0, 1000.0]),
        ambient_temp=np.array([15.0, 15.0, 15.0]),
        wind_speed=np.array([10.0, 1.0, 1.0]),
        wind_angle=np.array(
            [
                90.0,
                90.0,
                90.0,
            ]
        ),
    )
    return thermal_engine


def test_thermohl_cable_temp_arrays(cable_array_AM600: CableArray):
    thermal_engine = ThermalEngine()

    thermal_engine.set(
        cable_array_AM600,
        latitude=np.array([45.0, 44.0]),
        longitude=np.array([0.0, 0.0]),
        altitude=np.array([0.0, 0.0]),
        azimuth=np.array([0.0, 0.0]),
        month=np.array(
            [
                3,
                3,
            ]
        ),
        day=np.array(
            [
                21,
                21,
            ]
        ),
        hour=np.array(
            [
                12,
                12,
            ]
        ),
        intensity=np.array([100.0, 100.0]),
        ambient_temp=np.array([15.0, 15.0]),
        wind_speed=np.array([0.0, 10.0]),
        wind_angle=np.array(
            [
                90.0,
                90.0,
            ]
        ),
    )

    assert thermal_engine.steady_intensity().data.shape[0] == 2

    thermal_engine.set(
        cable_array_AM600,
        latitude=np.array([45.0, 45.0]),
        longitude=np.array([0.0, 0.0]),
        altitude=np.array([0.0, 0.0]),
        azimuth=np.array([0.0, 0.0]),
        month=np.array(
            [
                3,
                3,
            ]
        ),
        day=np.array(
            [
                21,
                21,
            ]
        ),
        hour=np.array(
            [
                12,
                12,
            ]
        ),
        intensity=np.array([100.0, 100.0]),
        ambient_temp=np.array([15.0, 15.0]),
        wind_speed=np.array([0.0, 0.0]),
        wind_angle=np.array(
            [
                90.0,
                90.0,
            ]
        ),
    )
    # expected 2 output rows, got 1 thl issue
    assert thermal_engine.steady_intensity().data.shape[0] == 1


def test_steady_intensity(thermal_engine_3_spans: ThermalEngine):
    thermal_engine = thermal_engine_3_spans

    copy_result_without_input = thermal_engine.steady_intensity().data.copy()

    assert thermal_engine.steady_intensity().data.shape[0] == 3

    np.testing.assert_array_almost_equal(
        copy_result_without_input,
        thermal_engine.steady_intensity(
            thermal_engine.target_temperature
        ).data,
        decimal=5,
    )

    assert (
        thermal_engine.steady_intensity(
            target_temperature=thermal_engine.target_temperature + 10
        ).data["t_core"]
        > copy_result_without_input["t_core"]
    ).all()


def test_steady_temperature(thermal_engine_3_spans: ThermalEngine):
    thermal_engine = thermal_engine_3_spans

    thermal_engine.dict_input["I"] = np.array([100.0, 200.0, 300.0])
    thermal_engine.load()

    copy_result_without_input = thermal_engine.steady_temperature().data.copy()

    assert thermal_engine.steady_temperature().data.shape[0] == 3

    # Testing manual input + changing just one parameter
    assert (
        copy_result_without_input["t_core"]
        != thermal_engine.steady_temperature(
            intensity=np.array([1000.0, 200.0, 300.0])
        ).data["t_core"]
    ).any()

    # testing higher intensity leads to higher temperature
    assert (
        thermal_engine.steady_temperature(
            intensity=np.array([1100.0, 1200.0, 1300.0])
        ).data["t_core"]
        > copy_result_without_input["t_core"]
    ).all()


def test_wrong_array_length(cable_array_AM600: CableArray):
    thermal_engine = ThermalEngine()
    with pytest.raises(
        ValueError,
        match="All array inputs must have the same length.",
    ):
        thermal_engine.set(
            cable_array_AM600,
            latitude=np.array([45.0, 45.0]),
            longitude=np.array([0.0, 0.0]),
            altitude=np.array([0.0, 0.0]),
            azimuth=np.array([0.0, 0.0]),
            month=np.array(
                [
                    3,
                    3,
                    3,
                ]
            ),
            day=np.array(
                [
                    21,
                    21,
                ]
            ),
            hour=np.array(
                [
                    12,
                    12,
                    12,
                ]
            ),
            intensity=np.array([100.0, 100.0]),
            ambient_temp=np.array([15.0, 15.0]),
            wind_speed=np.array([10.0, 10.0]),
            wind_angle=np.array(
                [
                    90.0,
                    90.0,
                ]
            ),
        )


def test_wrong_array_length_at_load(thermal_engine_3_spans: ThermalEngine):
    thermal_engine = thermal_engine_3_spans
    with pytest.raises(
        ValueError,
        match="All array inputs must have the same length.",
    ):
        thermal_engine.dict_input["latitude"] = np.array([45.0, 45.0])
        thermal_engine.load()


def test_add_manual_value_and_load(thermal_engine_3_spans: ThermalEngine):
    thermal_engine = thermal_engine_3_spans
    # latitude_old = thermal_engine.dict_input["lat"]

    thermal_engine.dict_input["lat"] = 40.0

    with pytest.raises(TypeError):
        thermal_engine.load()


def test_change_manual_value_and_load(thermal_engine_3_spans: ThermalEngine):
    thermal_engine = thermal_engine_3_spans
    latitude_old = thermal_engine.dict_input["lat"]

    thermal_engine.dict_input["lat"] = np.array([40.0, 40.0, 40.0])
    thermal_engine.load()

    assert not np.array_equal(
        thermal_engine.thermal_model.args.lat, latitude_old
    )


def test_len_str_repr(thermal_engine_3_spans: ThermalEngine):
    thermal_engine = thermal_engine_3_spans
    assert len(thermal_engine) == 3
    assert isinstance(str(thermal_engine), str)
    assert isinstance(repr(thermal_engine), str)

    thermal_results = thermal_engine.steady_temperature()
    assert len(thermal_results) == 3
    assert isinstance(str(thermal_results), str)
    assert isinstance(repr(thermal_results), str)


def test_transient_thermal(cable_array_AM600: CableArray):
    thermal_engine = ThermalEngine()
    thermal_engine.set(
        cable_array_AM600,
        latitude=np.array([45.0, 45.0, 45.0]),
        longitude=np.array([0.0, 0.0, 0.0]),
        altitude=np.array([0.0, 0.0, 0.0]),
        azimuth=np.array([0.0, 0.0, 20.0]),
        month=np.array(
            [
                3,
                3,
                3,
            ]
        ),
        day=np.array(
            [
                21,
                21,
                21,
            ]
        ),
        hour=np.array(
            [
                12,
                12,
                12,
            ]
        ),
        intensity=np.array([100.0, 100.0, 100.0]),
        ambient_temp=np.array([15.0, 15.0, 15.0]),
        wind_speed=np.array([10.0, 10.0, 0.0]),
        wind_angle=np.array(
            [
                90.0,
                80.0,
                90.0,
            ]
        ),
    )
    assert thermal_engine.transient_temperature().data.shape[0] == 3 * 10

    np.testing.assert_array_almost_equal(
        thermal_engine.wind_cable_angle, np.array([90.0, 80.0, 70.0])
    )


def test_steady_tempeature(thermal_engine_3_spans: ThermalEngine):
    steady_temp_results = thermal_engine_3_spans.steady_temperature()
    assert len(steady_temp_results.data) == 3

    np.testing.assert_array_almost_equal(
        steady_temp_results.data["t_core"],
        np.array([15.1, 45.4, 87.9]),
        decimal=0,
    )


def test_transient_results_raise_for_df_input():
    df_input = pd.DataFrame(
        {
            "time": [0, 1, 2],
            "id": [100, 150, 200],
            "t_avg": [15, 16, 17],
            "t_surf": [5, 10, 15],
            "t_core": [90, 80, 70],
        }
    )
    with pytest.raises(
        TypeError,
        match="DataFrame input not supported for transient results parsing.",
    ):
        ThermalTransientResults.parse_results(df_input)
