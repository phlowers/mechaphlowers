# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytest
from pytest import fixture

from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.data.catalog.catalog import sample_cable_catalog
from mechaphlowers.data.units import convert_weight_to_mass
from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
)
from mechaphlowers.entities.shapes import SupportShape
from mechaphlowers.plotting.plot import PlotEngine, plot_support_shape


@fixture
def balance_engine_local_test() -> BalanceEngine:
    data = {
        "name": ["1", "2", "three", "support 4"],
        "suspension": [False, True, True, False],
        "conductor_attachment_altitude": [50.0, 40.0, 20.0, 10.0],
        "crossarm_length": [
            5.0,
        ]
        * 4,
        "line_angle": [
            0,
        ]
        * 4,
        "insulator_length": [2, 4, 3.2, 2],
        "span_length": [100, 200, 300, np.nan],
        "insulator_mass": convert_weight_to_mass(
            np.array([1000.0, 500.0, 500.0, 1000.0])
        ),
        "load_mass": [0, 0, 0, 0],
        "load_position": [0, 0, 0, 0],
    }

    section = SectionArray(data=pd.DataFrame(data))
    section.sagging_parameter = 500
    section.sagging_temperature = 15
    cable_array_AM600: CableArray = sample_cable_catalog.get_as_object(
        ["ASTER600"]
    )  # type: ignore[assignment]

    balance_engine_local_test = BalanceEngine(
        cable_array=cable_array_AM600, section_array=section
    )
    balance_engine_local_test.solve_adjustment()
    return balance_engine_local_test


# frame = SectionDataFrame(section)


def test_plot_line3d__all_line(
    balance_engine_local_test: BalanceEngine,
) -> None:
    fig = go.Figure()
    plt_line = PlotEngine.builder_from_balance_engine(
        balance_engine_local_test
    )
    plt_line.preview_line3d(fig)
    # fig.show()  # deactivate for auto unit testing
    assert (
        len(
            [
                f
                for f in fig.data
                if f.name == "Cable" and not np.isnan(f.x).all()  # type: ignore[attr-defined]
            ]
        )
        == 1
    )
    assert len(fig.data) == 3  # Just trying to see if the previous code raises


def test_plot_line3d__view_option(
    balance_engine_local_test: BalanceEngine,
) -> None:
    fig = go.Figure()
    plt_line = PlotEngine.builder_from_balance_engine(
        balance_engine_local_test
    )
    plt_line.preview_line3d(fig, view="analysis")
    # fig.show()
    assert (
        len(
            [
                f
                for f in fig.data
                if f.name == "Cable" and not np.isnan(f.x).all()  # type: ignore[attr-defined]
            ]
        )
        == 1
    )
    assert len(fig.data) == 3  # Just trying to see if the previous code raises


def test_plot_line3d__wrong_view_option(
    balance_engine_local_test: BalanceEngine,
) -> None:
    fig = go.Figure()
    plt_line = PlotEngine.builder_from_balance_engine(
        balance_engine_local_test
    )
    plt_line.preview_line3d(fig)
    with pytest.raises(ValueError):
        plt_line.preview_line3d(fig, view="wrong_parameter")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        plt_line.preview_line3d(fig, view=22)  # type: ignore[arg-type]


def test_plot_line3d__with_beta(balance_engine_local_test: BalanceEngine):
    balance_engine_local_test.solve_change_state(
        ice_thickness=np.array([0.0, 60.0, 0.0, np.nan]),
        wind_pressure=np.array([240.12, 0.0, 600.0, np.nan]),
    )

    plt_line = PlotEngine.builder_from_balance_engine(
        balance_engine_local_test
    )
    fig = go.Figure()
    plt_line.preview_line3d(fig)
    # fig.show() # deactivate for auto unit testing
    assert (
        len(
            [
                f
                for f in fig.data
                if f.name == "Cable" and not np.isnan(f.x).all()  # type: ignore[attr-defined]
            ]
        )
        == 1
    )
    assert len(fig.data) == 3  # Just trying to see if the previous code raises


def test_plot_support_shape():
    fig = go.Figure()
    pyl_shape = SupportShape(
        name="pyl",
        yz_arms=np.array(
            [
                [0, 18.5],
                [3, 14.5],
                [6, 14.5],
                [9, 14.5],
                [-3, 14.5],
                [-6, 14.5],
                [-9, 14.5],
            ]
        ),
        set_number=np.array([22, 28, 37, 45, 46, 47, 55]),
    )
    plot_support_shape(fig, pyl_shape)
    # fig.show()
    assert True


def test_reactive_plot(balance_engine_base_test: BalanceEngine):
    plt_line = PlotEngine.builder_from_balance_engine(balance_engine_base_test)
    balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1, 1])
    )
    fig = go.Figure()
    plt_line.preview_line3d(fig)
    balance_engine_base_test.solve_change_state(
        wind_pressure=1000 * np.array([1, 1, 1, np.nan]),
    )

    plt_line.preview_line3d(fig)

    assert (
        len(
            [
                f
                for f in fig.data
                if f.name == "Cable" and not np.isnan(f.x).all()  # type: ignore[attr-defined]
            ]
        )
        == 2
    )

    # fig.show()  # deactivate for auto unit testing


def test_plot_ice(balance_engine_base_test: BalanceEngine):
    plt_line = PlotEngine.builder_from_balance_engine(balance_engine_base_test)
    balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1, 1])
    )
    plt_line = PlotEngine.builder_from_balance_engine(balance_engine_base_test)
    fig = go.Figure()
    plt_line.preview_line3d(fig)
    # balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(
        ice_thickness=np.array([1, 1, 1, np.nan]) * 1,
        # new_temperature=90 * np.array([1, 1, 1]),
    )

    plt_line = PlotEngine.builder_from_balance_engine(balance_engine_base_test)

    plt_line.preview_line3d(fig)

    # fig.show()  # deactivate for auto unit testing
    assert (
        len(
            [
                f
                for f in fig.data
                if f.name == "Cable" and not np.isnan(f.x).all()  # type: ignore[attr-defined]
            ]
        )
        == 2
    )
