# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytest
from pytest import fixture

from mechaphlowers.core.geometry.points import (
    Points,
    SectionPoints,
    stack_nan,
    vectors_to_coords,
)
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.core.models.cable.span import CatenarySpan
from mechaphlowers.entities.arrays import ObstacleArray, SectionArray
from mechaphlowers.plotting.plot import PlotEngine


@fixture
def section_array_line_angles():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["support 1", "2", "three", "support 4"]),
                "suspension": np.array([False, True, True, False]),
                "conductor_attachment_altitude": np.array([30, 40, 60, 70]),
                "crossarm_length": np.array([40, 20, -30, -50]),
                "line_angle": np.array([0, -45, 60, -30]),
                "insulator_length": np.array([0, 5, 82, 0]),
                "span_length": np.array([500, 460, 520, np.nan]),
                "insulator_mass": [1000.0, 500.0, 500.0, 1000.0],
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
    section_array.add_units({"line_angle": "deg"})
    return section_array


def test_stack_nan():
    """Test the stack_nan function."""

    coords = np.array(
        [
            [
                [-2.34427201e02, 0.00000000e00, 1.37547654e01],
                [-1.49742252e02, 0.00000000e00, 5.60830465e00],
                [-6.50573031e01, 0.00000000e00, 1.05820648e00],
                [1.96276459e01, 0.00000000e00, 9.63118943e-02],
                [1.04312595e02, 0.00000000e00, 2.72089608e00],
                [1.88997544e02, 0.00000000e00, 8.93666533e00],
                [2.73682493e02, 0.00000000e00, 1.87547654e01],
            ],
            [
                [-4.76437543e02, 0.00000000e00, 5.70170544e01],
                [-3.95654723e02, 0.00000000e00, 3.92634649e01],
                [-3.14871902e02, 0.00000000e00, 2.48373168e01],
                [-2.34089081e02, 0.00000000e00, 1.37150711e01],
                [-1.53306260e02, 0.00000000e00, 5.87857990e00],
                [-7.25234393e01, 0.00000000e00, 1.31505640e00],
                [8.25938147e00, 0.00000000e00, 1.70543698e-02],
            ],
            [
                [-4.76437543e02, 0.00000000e00, 5.70170544e01],
                [-3.95654723e02, 0.00000000e00, 3.92634649e01],
                [-3.14871902e02, 0.00000000e00, 2.48373168e01],
                [-2.34089081e02, 0.00000000e00, 1.37150711e01],
                [-1.53306260e02, 0.00000000e00, 5.87857990e00],
                [-7.25234393e01, 0.00000000e00, 1.31505640e00],
                [8.25938147e00, 0.00000000e00, 1.70543698e-02],
            ],
            [
                [8.80823413e01, 0.00000000e00, 1.93993824e00],
                [1.75171637e02, 0.00000000e00, 7.67618085e00],
                [2.62260932e02, 0.00000000e00, 1.72198528e01],
                [3.49350227e02, 0.00000000e00, 3.05890530e01],
                [4.36439522e02, 0.00000000e00, 4.78091353e01],
                [5.23528817e02, 0.00000000e00, 6.89127565e01],
                [6.10618113e02, 0.00000000e00, 9.39399382e01],
            ],
        ]
    )

    expected_output = np.array(
        [
            [-2.34427201e02, 0.00000000e00, 1.37547654e01],
            [-1.49742252e02, 0.00000000e00, 5.60830465e00],
            [-6.50573031e01, 0.00000000e00, 1.05820648e00],
            [1.96276459e01, 0.00000000e00, 9.63118943e-02],
            [1.04312595e02, 0.00000000e00, 2.72089608e00],
            [1.88997544e02, 0.00000000e00, 8.93666533e00],
            [2.73682493e02, 0.00000000e00, 1.87547654e01],
            [np.nan, np.nan, np.nan],
            [-4.76437543e02, 0.00000000e00, 5.70170544e01],
            [-3.95654723e02, 0.00000000e00, 3.92634649e01],
            [-3.14871902e02, 0.00000000e00, 2.48373168e01],
            [-2.34089081e02, 0.00000000e00, 1.37150711e01],
            [-1.53306260e02, 0.00000000e00, 5.87857990e00],
            [-7.25234393e01, 0.00000000e00, 1.31505640e00],
            [8.25938147e00, 0.00000000e00, 1.70543698e-02],
            [np.nan, np.nan, np.nan],
            [-4.76437543e02, 0.00000000e00, 5.70170544e01],
            [-3.95654723e02, 0.00000000e00, 3.92634649e01],
            [-3.14871902e02, 0.00000000e00, 2.48373168e01],
            [-2.34089081e02, 0.00000000e00, 1.37150711e01],
            [-1.53306260e02, 0.00000000e00, 5.87857990e00],
            [-7.25234393e01, 0.00000000e00, 1.31505640e00],
            [8.25938147e00, 0.00000000e00, 1.70543698e-02],
            [np.nan, np.nan, np.nan],
            [8.80823413e01, 0.00000000e00, 1.93993824e00],
            [1.75171637e02, 0.00000000e00, 7.67618085e00],
            [2.62260932e02, 0.00000000e00, 1.72198528e01],
            [3.49350227e02, 0.00000000e00, 3.05890530e01],
            [4.36439522e02, 0.00000000e00, 4.78091353e01],
            [5.23528817e02, 0.00000000e00, 6.89127565e01],
            [6.10618113e02, 0.00000000e00, 9.39399382e01],
            [np.nan, np.nan, np.nan],
        ]
    )

    result = stack_nan(coords)
    assert result.shape[1] == coords.shape[2]
    assert result.shape[0] == (coords.shape[1] + 1) * coords.shape[0]
    np.testing.assert_allclose(result, expected_output)


def test_point_class():
    """Test the Point class."""

    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]).T
    y = np.array([[0.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0]]).T
    z = np.array([[10.0, 20.0, 30.0, 40.0], [40.0, 30.0, 20.0, 10.0]]).T

    coords = vectors_to_coords(x, y, z)
    p1 = Points.from_coords(coords)

    x_p1, y_p1, z_p1 = p1.vectors

    np.testing.assert_almost_equal(x_p1, x)
    np.testing.assert_almost_equal(y_p1, y)
    np.testing.assert_almost_equal(z_p1, z)

    assert p1.points().shape == (2 * x.shape[0], 3)
    assert p1.points(stack=True).shape == (2 * y.shape[0] + 2, 3)

    p2 = Points.from_vectors(x, y, z)
    np.testing.assert_almost_equal(p1.coords, p2.coords)


@pytest.mark.skip(reason="To be fixed later // P1")
def test_span_absolute_coords_new_obj(section_array_line_angles):
    span_model = CatenarySpan(**section_array_line_angles.to_numpy())

    s = SectionPoints(
        span_model=span_model, **section_array_line_angles.to_numpy()
    )

    s.get_spans("cable").points(True)
    s.get_spans("localsection").points(True)
    s.get_spans("section").points(True)
    s.get_supports().points(True)
    s.get_insulators().points(True)

    s.init_span(span_model)

    # from plotly import graph_objects as go
    # from mechaphlowers.plotting.plot import plot_points_3d, set_layout

    # fig = go.Figure()
    # plot_points_3d(fig, s.get_spans("cable").points(True), name="Cable frame")
    # plot_points_3d(fig, s.get_spans("localsection").points(True), name="Localsection frame")
    # plot_points_3d(fig, s.get_spans("section").points(True), name="Section frame")

    # plot_points_3d(fig, s.get_supports().points(True), name="Supports")
    # plot_points_3d(fig, s.get_insulators().points(True), name="Insulators")

    # set_layout(fig)
    # fig.show()


@pytest.mark.skip(reason="To be fixed later // P1")
def test_span_with_wind(section_array_line_angles):
    span_model = CatenarySpan(**section_array_line_angles.to_numpy())

    s = SectionPoints(
        span_model=span_model, **section_array_line_angles.to_numpy()
    )
    s.beta = np.array([45, -45, 60, -60])
    points_cable_frame = s.get_spans("cable").points(True)
    y_spans = points_cable_frame[:, 1]
    y_spans_no_nans = y_spans[~np.isnan(y_spans)]
    # from plotly import graph_objects as go
    # from mechaphlowers.plotting.plot import plot_points_3d, set_layout

    # fig = go.Figure()
    # plot_points_3d(fig, s.get_spans("cable").points(True), name="Cable frame")
    # plot_points_3d(fig, s.get_spans("localsection").points(True), name="Localsection frame")
    # plot_points_3d(fig, s.get_spans("section").points(True), name="Section frame")

    # plot_points_3d(fig, s.get_supports().points(True), name="Supports")
    # plot_points_3d(fig, s.get_insulators().points(True), name="Insulators")

    # set_layout(fig)
    # fig.show()

    # Check that the y-coordinates of the spans are not all the same: the wind is active
    assert ~((y_spans_no_nans == y_spans_no_nans[0]).all())


def test_get_points(balance_engine_angles: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(balance_engine_angles)
    balance_engine_angles.solve_adjustment()
    balance_engine_angles.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1, 1])
    )
    with pytest.raises(ValueError):
        plt_engine.section_pts.get_points_for_plot(True, frame_index=4)


def test_plot_obstacles_sandbox(balance_engine_angles: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(balance_engine_angles)
    balance_engine_angles.solve_adjustment()
    balance_engine_angles.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1, 1])
    )

    obs_array = ObstacleArray(
        pd.DataFrame(
            {
                "name": ["obs_0", "obs_0", "obs_1", "obs_1", "obs_1", "obs_2"],
                "point_index": [0, 1, 0, 1, 2, 0],
                "span_index": [0, 0, 1, 1, 1, 1],
                "x": [
                    100.0,
                    200.0,
                    100.0,
                    200.0,
                    300.0,
                    200.0,
                ],
                "y": [0.0, 10.0, 0.0, 10.0, 10.0, -20.0],
                "z": [0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
                "object_type": [
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                ],
            }
        )
    )
    plt_engine.add_obstacles(obs_array)
    fig = go.Figure()
    plt_engine.section_pts.get_obstacle_coords()
    plt_engine.section_pts.obstacles_points.dict_coords()
    plt_engine.preview_line3d(fig)
    # fig.show()
    assert True


def test_obstacles_coords(balance_engine_angles: BalanceEngine):
    plt_engine = PlotEngine.builder_from_balance_engine(balance_engine_angles)
    balance_engine_angles.solve_adjustment()
    balance_engine_angles.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1, 1])
    )

    obs_array = ObstacleArray(
        pd.DataFrame(
            {
                "name": ["obs_0", "obs_0", "obs_1", "obs_1", "obs_1", "obs_2"],
                "point_index": [0, 1, 0, 1, 2, 0],
                "span_index": [0, 0, 1, 1, 1, 1],
                "x": [
                    100.0,
                    200.0,
                    100.0,
                    200.0,
                    300.0,
                    200.0,
                ],
                "y": [0.0, 10.0, 0.0, 10.0, 10.0, -20.0],
                "z": [0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
                "object_type": [
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                    "ground",
                ],
            }
        )
    )
    plt_engine.add_obstacles(obs_array)
    plt_engine.section_pts.obstacles_points.points(False)
    plt_engine.section_pts.obstacles_points.points(True)
    plt_engine.section_pts.obstacles_points.dict_coords()
