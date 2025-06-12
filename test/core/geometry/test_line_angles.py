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
    compute_span_azimuth,
    get_elevation_diff_between_supports,
    get_attachment_coords,
    get_edge_arm_coords,
    get_span_lengths_between_supports,
    get_supports_coords,
    get_supports_ground_coords,
    get_supports_layer,
    layer_to_plot,
)
from mechaphlowers.core.models.cable.span import CatenarySpan
from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.core.geometry.references import (
    SectionPoints,
    cable_to_beta_plane,
    cable_to_crossarm_frame,
    spans_to_vector,
    translate_cable_to_support,
)
from mechaphlowers.plotting.plot import set_layout


def plot_points_3d(fig, points):
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers+lines',
            marker=dict(
                size=3,
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
                "insulator_length": np.array([0, 5, 82, 0]),
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
    new_altitude_diff = get_elevation_diff_between_supports(arm_coords)

    expected_span_length = np.array(
        [508.10969425, 484.69692488, 522.53577119, np.nan]
    )
    expected_altitude_diff = np.array([10, 20, 10, np.nan])
    np.testing.assert_allclose(new_span_length, expected_span_length)
    np.testing.assert_allclose(new_altitude_diff, expected_altitude_diff)


def test_get_supports_coords(section_array_line_angles):
    span_length = section_array_line_angles.data.span_length.to_numpy()
    line_angle = section_array_line_angles.data.line_angle.to_numpy()
    conductor_attachment_altitude = (
        section_array_line_angles.data.conductor_attachment_altitude.to_numpy()
    )
    crossarm_length = section_array_line_angles.data.crossarm_length.to_numpy()
    insulator_length = (
        section_array_line_angles.data.insulator_length.to_numpy()
    )

    ground_cds, center_arm_cds, arm_cds, attachment_cds = get_supports_coords(
        span_length,
        line_angle,
        conductor_attachment_altitude,
        crossarm_length,
        insulator_length,
    )
    assert True


def test_span_absolute_coords_new_obj(section_array_line_angles):
    # ---- Same code than previous test ----""
    span_length = section_array_line_angles.data.span_length.to_numpy()
    line_angle = section_array_line_angles.data.line_angle.to_numpy()
    conductor_attachment_altitude = (
        section_array_line_angles.data.conductor_attachment_altitude.to_numpy()
    )
    crossarm_length = section_array_line_angles.data.crossarm_length.to_numpy()
    insulator_length = (
        section_array_line_angles.data.insulator_length.to_numpy()
    )

    (
        supports_ground_coords,
        center_arm_coords,
        arm_coords,
        attachment_coords,
    ) = get_supports_coords(
        span_length,
        line_angle,
        conductor_attachment_altitude,
        crossarm_length,
        insulator_length,
    )

    new_span_length = get_span_lengths_between_supports(attachment_coords)
    new_elevation_diff = get_elevation_diff_between_supports(attachment_coords)
    # ----------

    span_model = CatenarySpan(**section_array_line_angles.to_numpy())
    span_model.span_length = new_span_length
    span_model.elevation_difference = new_elevation_diff

    s = SectionPoints(
        span_length,
        conductor_attachment_altitude,
        crossarm_length,
        insulator_length,
        line_angle,
    )

    s.init_span(span_model)

    fig = go.Figure()
    plot_points_3d(fig, s.get_spans("cable").points(True))
    plot_points_3d(fig, s.get_spans("crossarm").points(True))
    plot_points_3d(fig, s.get_spans("section").points(True))

    plot_points_3d(fig, s.get_supports().points(True))

    set_layout(fig)
    fig.show()
    assert True


def test_span_absolute_coords(section_array_line_angles):
    # ---- Same code than previous test ----""
    span_length = section_array_line_angles.data.span_length.to_numpy()
    line_angle = section_array_line_angles.data.line_angle.to_numpy()
    conductor_attachment_altitude = (
        section_array_line_angles.data.conductor_attachment_altitude.to_numpy()
    )
    crossarm_length = section_array_line_angles.data.crossarm_length.to_numpy()
    insulator_length = (
        section_array_line_angles.data.insulator_length.to_numpy()
    )

    (
        supports_ground_coords,
        center_arm_coords,
        arm_coords,
        attachment_coords,
    ) = get_supports_coords(
        span_length,
        line_angle,
        conductor_attachment_altitude,
        crossarm_length,
        insulator_length,
    )

    new_span_length = get_span_lengths_between_supports(attachment_coords)
    new_elevation_diff = get_elevation_diff_between_supports(attachment_coords)
    # ----------

    span_model = CatenarySpan(**section_array_line_angles.to_numpy())
    span_model.span_length = new_span_length
    span_model.elevation_difference = new_elevation_diff

    x = span_model.x(21)
    z = span_model.z(x)

    pts = spans_to_vector(x, x * 0, z)

    fig = go.Figure()
    # plot_points_3d(fig, pts)
    # fig.show()

    # get pts for spans

    beta = np.array([50.0, 10.0, 0.0, np.nan])
    x_span, y_span, z_span = cable_to_beta_plane(
        x[:, :-1], z[:, :-1], beta=beta[:-1]
    )

    pts = spans_to_vector(x_span, y_span, z_span)

    # just to verify cable frame
    plot_points_3d(fig, pts)

    alpha = compute_span_azimuth(attachment_coords)
    x_span, y_span, z_span = cable_to_crossarm_frame(
        x_span, y_span, z_span, alpha[:-1]
    )

    pts = spans_to_vector(x_span, y_span, z_span)
    # just to verify spans in crossarm frame
    plot_points_3d(fig, pts)

    x_span, y_span, z_span = translate_cable_to_support(
        x_span,
        y_span,
        z_span,
        conductor_attachment_altitude,
        span_length,
        crossarm_length,
        insulator_length,
        line_angle,
    )

    pts = spans_to_vector(x_span, y_span, z_span)
    # final span plot
    plot_points_3d(fig, pts)

    supports_layer = get_supports_layer(
        supports_ground_coords,
        center_arm_coords,
        arm_coords,
        attachment_coords,
    )
    supports_coords = layer_to_plot(supports_layer)
    plot_points_3d(fig, supports_coords)

    set_layout(fig)

    fig.show()
    fig.to_html()

    assert True


def test_span_absolute_coords_(section_array_line_angles):
    # ---- Same code than previous test ----""
    span_length = section_array_line_angles.data.span_length.to_numpy()
    line_angle = section_array_line_angles.data.line_angle.to_numpy()
    conductor_attachment_altitude = (
        section_array_line_angles.data.conductor_attachment_altitude.to_numpy()
    )
    crossarm_length = section_array_line_angles.data.crossarm_length.to_numpy()
    insulator_length = (
        section_array_line_angles.data.insulator_length.to_numpy()
    )

    (
        supports_ground_coords,
        center_arm_coords,
        arm_coords,
        attachment_coords,
    ) = get_supports_coords(
        span_length,
        line_angle,
        conductor_attachment_altitude,
        crossarm_length,
        insulator_length,
    )

    new_span_length = get_span_lengths_between_supports(attachment_coords)
    new_elevation_diff = get_elevation_diff_between_supports(attachment_coords)
    # ----------

    span_model = CatenarySpan(**section_array_line_angles.to_numpy())
    span_model.span_length = new_span_length
    span_model.elevation_difference = new_elevation_diff

    x = span_model.x(21)
    z = span_model.z(x)

    pts = spans_to_vector(x, x * 0, z)

    fig = go.Figure()
    # plot_points_3d(fig, pts)
    # fig.show()

    # get pts for spans

    beta = np.array([50.0, 10.0, 0.0, np.nan])
    x_span, y_span, z_span = cable_to_beta_plane(
        x[:, :-1], z[:, :-1], beta=beta[:-1]
    )

    pts = spans_to_vector(x_span, y_span, z_span)

    # just to verify cable frame
    plot_points_3d(fig, pts)

    alpha = compute_span_azimuth(attachment_coords)
    x_span, y_span, z_span = cable_to_crossarm_frame(
        x_span, y_span, z_span, alpha[:-1]
    )

    pts = spans_to_vector(x_span, y_span, z_span)
    # just to verify spans in crossarm frame
    plot_points_3d(fig, pts)

    x_span, y_span, z_span = translate_cable_to_support(
        x_span,
        y_span,
        z_span,
        conductor_attachment_altitude,
        span_length,
        crossarm_length,
        insulator_length,
        line_angle,
    )

    pts = spans_to_vector(x_span, y_span, z_span)
    # final span plot
    plot_points_3d(fig, pts)

    supports_layer = get_supports_layer(
        supports_ground_coords,
        center_arm_coords,
        arm_coords,
        attachment_coords,
    )
    supports_coords = layer_to_plot(supports_layer)
    plot_points_3d(fig, supports_coords)

    set_layout(fig)

    fig.show()
    fig.to_html()

    assert True
