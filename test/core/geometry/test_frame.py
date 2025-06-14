# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from numpy.testing import assert_allclose
from pytest import fixture

from mechaphlowers.core.geometry.frame import Frame, Points


def plot_points(fig, points):
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers+lines',
            marker=dict(
                size=4,
            ),  # color='red'),
            name='Points',
        )
    )


def plot_frame(fig, frame, one_object=False, color='blue'):
    pp = np.hstack(
        (
            frame.origin,
            frame.origin + frame.x_axis,
            np.nan * np.ones_like(frame.origin),
            frame.origin,
            frame.origin + frame.y_axis,
            np.nan * np.ones_like(frame.origin),
            frame.origin,
            frame.origin + frame.z_axis,
            np.nan * np.ones_like(frame.origin),
        )
    )
    if one_object:
        pp = pp.reshape(-1, 3)
        fig.add_trace(
            go.Scatter3d(
                x=pp[:, 0],
                y=pp[:, 1],
                z=pp[:, 2],
                mode='lines',
                marker=dict(size=5, color=color),
                name="Frame",
            )
        )
    else:
        for i in range(pp.shape[0]):
            fig.add_trace(
                go.Scatter3d(
                    x=pp[i].reshape(-1, 3)[:, 0],
                    y=pp[i].reshape(-1, 3)[:, 1],
                    z=pp[i].reshape(-1, 3)[:, 2],
                    mode='lines',
                    marker=dict(size=5, color=color),
                    name=f"R{str(i)}",
                )
            )
    return fig


@fixture
def coords_fixture():
    return np.array(
        [
            [100.0, -40.0, 30.0],
            [183.33333333, -40.0, 22.13841697],
            [266.66666667, -40.0, 17.75890716],
            [350.0, -40.0, 16.85386616],
            [433.33333333, -40.0, 19.42172249],
            [516.66666667, -40.0, 25.46693488],
            [600.0, -40.0, 35.0],
            [100.0, -10.0, 30.0],
            [183.33333333, -10.0, 22.13841697],
            [266.66666667, -10.0, 17.75890716],
            [350.0, -10.0, 16.85386616],
            [433.33333333, -10.0, 19.42172249],
            [516.66666667, -10.0, 25.46693488],
            [600.0, -10.0, 35.0],
            [500.0, -40.0, 35.0],
            [583.33333333, -40.0, 28.80643923],
            [666.66666667, -40.0, 26.0905585],
            [750.0, -40.0, 26.84764206],
            [833.33333333, -40.0, 31.07900449],
            [916.66666667, -40.0, 38.79199295],
            [1000.0, -40.0, 50.0],
            [1000.0, 460.0, 50.0],
            [1083.33333333, 460.0, 44.63915405],
            [1166.66666667, 460.0, 42.75431097],
            [1250.0, 460.0, 44.342198],
            [1333.33333333, 460.0, 49.40557229],
            [1416.66666667, 460.0, 57.95322568],
            [1500.0, 460.0, 70.0],
        ]
    ).reshape((4, 7, 3))


def test_plot(coords_fixture):
    fig = go.Figure()
    plot_points(fig, coords_fixture.reshape(-1, 3))

    # fig.show()


def test_translate_all(coords_fixture):
    expected_translated_coords = coords_fixture.copy()
    points = Points(coords_fixture)
    translation_vector = np.array([[0, 1, 0]] * 4)
    points.translate_all(translation_vector)

    translation = np.full(expected_translated_coords.shape, [0, 1, 0])
    expected_translated_coords += translation

    assert_allclose(points.coords, expected_translated_coords)


def test_translate_layer(coords_fixture):
    expected_translated_coords = coords_fixture.copy()
    points = Points(coords_fixture)
    translation_vector = np.array([0, 1, 0])
    points.translate_layer(translation_vector, 3)

    translation = np.full(expected_translated_coords[3].shape, [0, 1, 0])
    expected_translated_coords[3] += translation

    assert_allclose(points.coords, expected_translated_coords)


def test_flatten(coords_fixture):
    points = Points(coords_fixture)
    assert points.flatten().shape == (28, 3)


def test_rotate_layer(coords_fixture):
    fig = go.Figure()
    plot_points(fig, coords_fixture.reshape(-1, 3))
    points = Points(coords_fixture)
    line_angles = np.array([180, 90, 180, 0])
    rotation_axes = np.array([[0, 0, 1]] * 4)
    points.rotate_all(line_angles, rotation_axes)

    plot_points(fig, points.coords_for_plot())

    # fig.show()


def test_rotate_point(coords_fixture):
    fig = go.Figure()
    plot_points(fig, coords_fixture.reshape(-1, 3))
    points = Points(coords_fixture)
    line_angles = np.array([20] * points.flatten().shape[0])
    rotation_axes = np.array([[0, 0, 1]] * 4)
    points.rotate_one_angle_per_point(line_angles, rotation_axes)

    plot_points(fig, points.coords_for_plot())

    # fig.show()


def test_rotate_point_same_axis(coords_fixture):
    fig = go.Figure()
    plot_points(fig, coords_fixture.reshape(-1, 3))
    points = Points(coords_fixture)
    line_angles = np.array([180, 90, 180, 0])
    rotation_axis = np.array([0, 0, 1])
    points.rotate_same_axis(line_angles, rotation_axis)

    plot_points(fig, points.coords_for_plot())

    # fig.show()


def test_plot_frame(coords_fixture):
    frame = Frame(np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 1, 1]]))

    fig = go.Figure()

    plot_frame(fig, frame)

    # fig.show()


def test_points_rotate_frame(coords_fixture):
    frame = Frame(np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 1, 1]]))
    fig = go.Figure()
    plot_points(fig, coords_fixture.reshape(-1, 3))
    plot_frame(fig, frame)
    points = Points(coords_fixture, frame)
    line_angles = np.array([180, 90, 45, 0])
    rotation_axes = np.array([[0, 0, 1]] * 4)
    points.rotate_all(line_angles, rotation_axes)

    points.rotate_frame(-line_angles, rotation_axes)
    plot_points(fig, points.coords_for_plot())
    plot_frame(fig, points.frame, color='red')
    # fig.show()


def test_points_translate_frame(coords_fixture):
    frame = Frame(np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 1, 1]]))

    # frame = Frame(np.array([[0.,0.,0.], [1.,1.,1.], [1.,1.,1.], [2.,1.,1.]]))
    fig = go.Figure()
    plot_points(fig, coords_fixture.reshape(-1, 3))
    plot_frame(fig, frame)
    points = Points(coords_fixture, frame)
    translation_vector = np.array(
        [[50, 0, 0], [0, 50, 0], [0, 0, 50], [25, 25, 0]]
    )
    points.translate_all(translation_vector)

    points.translate_frame(-translation_vector)
    plot_points(fig, points.coords_for_plot())
    plot_frame(fig, points.frame, color='red')
    # fig.show()
