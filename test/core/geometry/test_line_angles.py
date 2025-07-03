# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pytest import fixture

from mechaphlowers.core.geometry.line_angles import (
    angle_between_vectors,
    compute_span_azimuth,
    get_attachment_coords,
    get_edge_arm_coords,
    get_elevation_diff_between_supports,
    get_span_lengths_between_supports,
    get_supports_coords,
    get_supports_ground_coords,
    get_supports_layer,
)
from mechaphlowers.core.geometry.points import coords_to_points
from mechaphlowers.entities.arrays import SectionArray


@fixture
def section_array_line_angles():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["support 1", "2", "three", "support 4"]),
                "suspension": np.array([False, True, True, False]),
                "conductor_attachment_altitude": np.array([30, 35, -22, 70]),
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


def test_get_supports_ground_coords(section_array_line_angles):
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

    # import plotly.graph_objects as go
    # from mechaphlowers.plotting.plot import plot_points_3d
    # fig = go.Figure()
    # plot_points_3d(fig, supports_ground_coords)
    # fig.show()


def test_build_supports(section_array_line_angles):
    span_length = section_array_line_angles.data.span_length.to_numpy()
    line_angle = section_array_line_angles.data.line_angle.to_numpy()
    insulator_length = (
        section_array_line_angles.data.insulator_length.to_numpy()
    )
    conductor_attachment_altitude = (
        section_array_line_angles.data.conductor_attachment_altitude.to_numpy()
    )
    crossarm_length = section_array_line_angles.data.crossarm_length.to_numpy()

    supports_ground_coords = get_supports_ground_coords(
        span_length, line_angle
    )
    center_arm_coords, edge_arm_coords = get_edge_arm_coords(
        supports_ground_coords,
        conductor_attachment_altitude,
        crossarm_length,
        line_angle,
        insulator_length,
    )

    expected_center_arm = np.array(
        [
            [0.0, 0.0, 30.0],
            [500.0, 0.0, 40.0],
            [825.26911935, -325.26911935, 60.0],
            [1327.55054902, -190.68321589, 70.0],
        ]
    )

    expected_edge_arm = np.array(
        [
            [0.0, 40.0, 30.0],
            [507.65366865, 18.47759065, 40.0],
            [817.50454799, -354.24689413, 60.0],
            [1327.55054902, -240.68321589, 70.0],
        ]
    )

    np.testing.assert_allclose(center_arm_coords, expected_center_arm)
    np.testing.assert_allclose(edge_arm_coords, expected_edge_arm)

    # import plotly.graph_objects as go
    # from mechaphlowers.plotting.plot import plot_points_3d
    # fig = go.Figure()
    # plot_points_3d(fig, supports_ground_coords)
    # plot_points_3d(fig, center_arm_coords)
    # plot_points_3d(fig, edge_arm_coords)
    # fig.show()


def test_build_attachments(section_array_line_angles):
    span_length = section_array_line_angles.data.span_length.to_numpy()
    line_angle = section_array_line_angles.data.line_angle.to_numpy()
    insulator_length = (
        section_array_line_angles.data.insulator_length.to_numpy()
    )
    conductor_attachment_altitude = (
        section_array_line_angles.data.conductor_attachment_altitude.to_numpy()
    )
    crossarm_length = section_array_line_angles.data.crossarm_length.to_numpy()

    supports_ground_coords = get_supports_ground_coords(
        span_length, line_angle
    )
    center_arm_coords, edge_arm_coords = get_edge_arm_coords(
        supports_ground_coords,
        conductor_attachment_altitude,
        crossarm_length,
        line_angle,
        insulator_length,
    )

    attachment_coords = get_attachment_coords(
        edge_arm_coords, conductor_attachment_altitude
    )

    expected_attachment = np.array(
        [
            [0.0, 40.0, 30.0],
            [507.65366865, 18.47759065, 35.0],
            [817.50454799, -354.24689413, -22.0],
            [1327.55054902, -240.68321589, 70.0],
        ]
    )

    np.testing.assert_allclose(attachment_coords, expected_attachment)
    # import plotly.graph_objects as go
    # from mechaphlowers.plotting.plot import plot_points_3d
    # fig = go.Figure()
    # plot_points_3d(fig, supports_ground_coords)
    # plot_points_3d(fig, center_arm_coords)
    # plot_points_3d(fig, edge_arm_coords)
    # plot_points_3d(fig, attachment_coords)
    # fig.show()


def test_supports_coords_to_points(section_array_line_angles):
    """Get the supports in the support frame."""
    span_length = section_array_line_angles.data.span_length.to_numpy()
    line_angle = section_array_line_angles.data.line_angle.to_numpy()
    insulator_length = (
        section_array_line_angles.data.insulator_length.to_numpy()
    )
    conductor_attachment_altitude = (
        section_array_line_angles.data.conductor_attachment_altitude.to_numpy()
    )
    crossarm_length = section_array_line_angles.data.crossarm_length.to_numpy()

    supports_ground_coords = get_supports_ground_coords(
        span_length, line_angle
    )
    center_arm_coords, edge_arm_coords = get_edge_arm_coords(
        supports_ground_coords,
        conductor_attachment_altitude,
        crossarm_length,
        line_angle,
        insulator_length,
    )

    supports_layer = get_supports_layer(
        supports_ground_coords,
        center_arm_coords,
        edge_arm_coords,
    )
    supports_coords = coords_to_points(supports_layer)

    expected_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [500.0, 0.0, 0.0],
            [825.26911935, -325.26911935, 0.0],
            [1327.55054902, -190.68321589, 0.0],
            [0.0, 0.0, 30.0],
            [500.0, 0.0, 40.0],
            [825.26911935, -325.26911935, 60.0],
            [1327.55054902, -190.68321589, 70.0],
            [0.0, 40.0, 30.0],
            [507.65366865, 18.47759065, 40.0],
            [817.50454799, -354.24689413, 60.0],
            [1327.55054902, -240.68321589, 70.0],
        ]
    )

    assert_allclose(supports_coords, expected_coords)


def test_span_lengths(section_array_line_angles):
    span_length = section_array_line_angles.data.span_length.to_numpy()
    line_angle = section_array_line_angles.data.line_angle.to_numpy()
    insulator_length = (
        section_array_line_angles.data.insulator_length.to_numpy()
    )

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
        crossarm_length,
        line_angle,
        insulator_length,
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


def test_compute_span_azimuth():
    attachement_coords = np.array(
        [
            [0.0, 40.0, 30.0],
            [507.65366865, 18.47759065, 35.0],
            [817.50454799, -354.24689413, -22.0],
            [1327.55054902, -240.68321589, 70.0],
        ]
    )

    expected_azimuth = np.array(
        [-2.42764954, -50.26279675, 12.55236927, np.nan]
    )

    np.testing.assert_allclose(
        compute_span_azimuth(attachement_coords), expected_azimuth
    )


def test_angle_between_vectors():
    # Define input vectors
    vector_a = np.array([[1, 0], [0, 1]])
    vector_b = np.array([[0, 1], [1, 0]])

    # Call the function
    result = angle_between_vectors(vector_a, vector_b)

    # Define the expected result
    expected_result = np.array([np.pi / 2, -np.pi / 2])

    # Assert that the result matches the expected result
    np.testing.assert_array_almost_equal(result, expected_result, decimal=5)


def test_angle_between_vectors_above_90():
    # Define input vectors
    vector_a = np.array([[1, 0], [-1, 1]])
    vector_b = np.array([[-1, 1], [1, 0]])

    # Call the function
    result = angle_between_vectors(vector_a, vector_b)

    # Define the expected result
    expected_result = np.array([3 * np.pi / 4, -3 * np.pi / 4])

    # Assert that the result matches the expected result
    np.testing.assert_array_almost_equal(result, expected_result, decimal=5)


def test_angle_between_vectors_parallel():
    # Define input vectors that are parallel
    vector_a = np.array([[1, 0], [0, 1]])
    vector_b = np.array([[2, 0], [0, 2]])

    # Call the function
    result = angle_between_vectors(vector_a, vector_b)

    # Define the expected result
    expected_result = np.array([0, 0])

    # Assert that the result matches the expected result
    np.testing.assert_array_almost_equal(result, expected_result, decimal=5)


def test_angle_between_vectors_antiparallel():
    # Define input vectors that are antiparallel
    vector_a = np.array([[1, 0], [0, 1]])
    vector_b = -vector_a

    # Call the function
    result = angle_between_vectors(vector_a, vector_b)

    # Define the expected result
    expected_result = np.array([np.pi, np.pi])

    # Assert that the result matches the expected result
    np.testing.assert_array_almost_equal(result, expected_result, decimal=5)


def test_angle_between_vectors_zero_vector():
    # Define input vectors with a zero vector
    vector_a = np.array([[1, 0], [0, 1], [0, 0]])
    vector_b = np.array([[0, 0], [0, 0], [0, 0]])

    # Call the function
    result = angle_between_vectors(vector_a, vector_b)
    expected_result = np.array([np.nan, np.nan, np.nan])

    np.testing.assert_array_almost_equal(result, expected_result, decimal=5)


def test_angle_between_vectors_same_vector():
    # Define input vectors that are the same
    vector_a = np.array([[1, 0], [0, 1], [0, 0]])
    vector_b = np.array([[1, 0], [0, 1], [0, 0]])

    # Call the function
    result = angle_between_vectors(vector_a, vector_b)

    # Define the expected result
    expected_result = np.array([0, 0, np.nan])

    # Assert that the result matches the expected result
    np.testing.assert_array_almost_equal(result, expected_result, decimal=5)
