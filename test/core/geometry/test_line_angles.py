# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.testing import assert_allclose
from pytest import fixture

from mechaphlowers.core.geometry.line_angles import (
    get_altitude_diff_between_supports,
    get_attachment_coords,
    get_edge_arm_coords,
    get_span_lengths_between_supports,
    get_supports_ground_coords,
    get_supports_layer,
    layer_to_plot,
)
from mechaphlowers.core.models.cable.span import CatenarySpan
from mechaphlowers.entities.arrays import SectionArray


def plot_points_3d(fig, points):
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers+lines',
            marker=dict(
                size=10,
            ),  # color='red'),
            name='Points',
        )
    )


def plot_points_2d(fig, points):
    fig.add_trace(
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            mode='markers+lines',
            marker=dict(
                size=4,
            ),  # color='red'),
            name='Points',
        )
    )


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
                "insulator_length": np.array([0, 5, 10, 0]),
                "span_length": np.array([500, 460, 520, np.nan]),
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15
    return section_array


def test_get_supports_coords(section_array_line_angles):
    span_length = section_array_line_angles.data.span_length.to_numpy()
    line_angle = section_array_line_angles.data.line_angle.to_numpy()
    supports_ground_coords = get_supports_ground_coords(
        span_length, line_angle
    )
    expected_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [500.0, 0.0, 0.0],
            [825.26911935, -325.26911935, 0.0],
            [1327.55054902, -190.68321589, 0.0],
        ]
    )
    np.testing.assert_allclose(supports_ground_coords, expected_coords)

    fig = go.Figure()
    plot_points_2d(fig, supports_ground_coords)


# fig.show()


def test_build_supports(section_array_line_angles):
    span_length = section_array_line_angles.data.span_length.to_numpy()
    line_angle = section_array_line_angles.data.line_angle.to_numpy()
    conductor_attachment_altitude = (
        section_array_line_angles.data.conductor_attachment_altitude.to_numpy()
    )
    crossarm_length = section_array_line_angles.data.crossarm_length.to_numpy()

    supports_ground_coords = get_supports_ground_coords(
        span_length, line_angle
    )
    center_arm_coords, arm_coords = get_edge_arm_coords(
        supports_ground_coords,
        conductor_attachment_altitude,
        line_angle,
        crossarm_length,
    )

    expected_coords = np.array(
        [
            [0.0, 40.0, 30.0],
            [507.65366865, 18.47759065, 40.0],
            [817.50454799, -354.24689413, 60.0],
            [1327.55054902, -240.68321589, 70.0],
        ]
    )

    np.testing.assert_allclose(arm_coords, expected_coords)

    # fig = go.Figure()
    # plot_points_3d(fig, supports_ground_coords)
    # plot_points_3d(fig, center_arm_coords)
    # plot_points_3d(fig, arm_coords)

    # fig.show()


def test_get_supports(section_array_line_angles):
    """Get the supports in the global frame."""
    span_length = section_array_line_angles.data.span_length.to_numpy()
    line_angle = section_array_line_angles.data.line_angle.to_numpy()
    conductor_attachment_altitude = (
        section_array_line_angles.data.conductor_attachment_altitude.to_numpy()
    )
    crossarm_length = section_array_line_angles.data.crossarm_length.to_numpy()
    insulator_length = (
        section_array_line_angles.data.insulator_length.to_numpy()
    )

    supports_ground_coords = get_supports_ground_coords(
        span_length, line_angle
    )
    center_arm_coords, edge_arm_coords = get_edge_arm_coords(
        supports_ground_coords,
        conductor_attachment_altitude,
        line_angle,
        crossarm_length,
    )
    attachment_coords = get_attachment_coords(
        edge_arm_coords, insulator_length
    )

    supports_layer = get_supports_layer(
        supports_ground_coords,
        center_arm_coords,
        edge_arm_coords,
        attachment_coords,
    )
    supports_coords = layer_to_plot(supports_layer)

    expected_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 30.0],
            [0.0, 40.0, 30.0],
            [0.0, 40.0, 30.0],
            [np.nan, np.nan, np.nan],
            [500.0, 0.0, 0.0],
            [500.0, 0.0, 40.0],
            [507.65366865, 18.47759065, 40.0],
            [507.65366865, 18.47759065, 35.0],
            [np.nan, np.nan, np.nan],
            [825.26911935, -325.26911935, 0.0],
            [825.26911935, -325.26911935, 60.0],
            [817.50454799, -354.24689413, 60.0],
            [817.50454799, -354.24689413, 50.0],
            [np.nan, np.nan, np.nan],
            [1327.55054902, -190.68321589, 0.0],
            [1327.55054902, -190.68321589, 70.0],
            [1327.55054902, -240.68321589, 70.0],
            [1327.55054902, -240.68321589, 70.0],
            [np.nan, np.nan, np.nan],
        ]
    )

    assert_allclose(supports_coords, expected_coords)

    # fig = go.Figure()
    # plot_points_3d(fig, supports_coords)
    # # set_layout(fig)
    # fig.show()


def test_span_lengths(section_array_line_angles):
    span_length = section_array_line_angles.data.span_length.to_numpy()
    line_angle = section_array_line_angles.data.line_angle.to_numpy()
    conductor_attachment_altitude = (
        section_array_line_angles.data.conductor_attachment_altitude.to_numpy()
    )
    crossarm_length = section_array_line_angles.data.crossarm_length.to_numpy()

    supports_ground_coords = get_supports_ground_coords(
        span_length, line_angle
    )
    center_arm_coords, arm_coords = get_edge_arm_coords(
        supports_ground_coords,
        conductor_attachment_altitude,
        line_angle,
        crossarm_length,
    )
    new_span_length = get_span_lengths_between_supports(arm_coords)
    new_altitude_diff = get_altitude_diff_between_supports(arm_coords)

    expected_span_length = np.array(
        [508.10969425, 484.69692488, 522.53577119, np.nan]
    )
    expected_altitude_diff = np.array([10, 20, 10, np.nan])
    np.testing.assert_allclose(new_span_length, expected_span_length)
    np.testing.assert_allclose(new_altitude_diff, expected_altitude_diff)


def test_span_absolute_coords(section_array_line_angles):
    # ---- Same code than previous test ----""
    span_length = section_array_line_angles.data.span_length.to_numpy()
    line_angle = section_array_line_angles.data.line_angle.to_numpy()
    conductor_attachment_altitude = (
        section_array_line_angles.data.conductor_attachment_altitude.to_numpy()
    )
    crossarm_length = section_array_line_angles.data.crossarm_length.to_numpy()

    supports_ground_coords = get_supports_ground_coords(
        span_length, line_angle
    )
    center_arm_coords, arm_coords = get_edge_arm_coords(
        supports_ground_coords,
        conductor_attachment_altitude,
        line_angle,
        crossarm_length,
    )
    new_span_length = get_span_lengths_between_supports(arm_coords)
    new_altitude_diff = get_altitude_diff_between_supports(arm_coords)
    # ----------

    span_model = CatenarySpan(**section_array_line_angles.to_numpy())
