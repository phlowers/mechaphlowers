# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.geometry.references import (
    cable_to_beta_plane,
    translate_cable_to_support,
    vectors_to_points,
)


# function defined many times in several files. Maybe refacto?
def create_default_displacement_vector(
    insulator_length: np.ndarray,
) -> np.ndarray:
    displacement_vector = np.zeros((insulator_length.size, 3))
    displacement_vector[1:-1, 2] = -insulator_length[1:-1]
    displacement_vector[0, 0] = insulator_length[0]
    displacement_vector[-1:, 0] = -insulator_length[-1]
    return displacement_vector


def test_cable2span_basic() -> None:
    x: np.ndarray = np.array([[1, 2, 3, 4], [10, 12, 14, 16]]).T
    z: np.ndarray = np.array([[20, 18, 17, 19], [19, 17, 15, 17]]).T
    beta: float = 0

    xs, ys, zs = cable_to_beta_plane(
        x, z, np.ones(2) * beta, np.array([3, 6]), np.array([-1, -2])
    )  # TODO check beta

    assert len(xs) == len(z)
    np.testing.assert_allclose(ys, np.zeros_like(ys))
    # assert np.allclose(result, z)
    xs, ys, zs = cable_to_beta_plane(
        x, z, np.array([5.0, 61.3]), np.array([3, 6]), np.array([-1, -2])
    )  # TODO check beta
    assert len(xs) == len(z)
    assert not (ys == 0).all()


def test_spans2vector() -> None:
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    z = np.array([[9, 10], [11, 12]])

    expected_output = np.array([[1, 5, 9], [3, 7, 11], [2, 6, 10], [4, 8, 12]])

    result = vectors_to_points(x, y, z)

    assert np.array_equal(
        result, expected_output
    ), f"Expected {expected_output}, but got {result}"


def test_spans2vector_empty() -> None:
    x = np.array([[]])
    y = np.array([[]])
    z = np.array([[]])

    # expected_output = np.array([[]]) so we check that size == 0

    result = vectors_to_points(x, y, z)

    assert result.size == 0


def test_spans2vector_single_point() -> None:
    x = np.array([[1]])
    y = np.array([[2]])
    z = np.array([[3]])

    expected_output = np.array([[1, 2, 3]])

    result = vectors_to_points(x, y, z)

    assert np.array_equal(
        result, expected_output
    ), f"Expected {expected_output}, but got {result}"


def test_translate_cable_to_support() -> None:
    altitude = np.array([48.0, 39.0, 19.0, 10.0])
    span_length = np.array([100.0, 200.0, 300.0, np.nan])
    crossarm_length = np.array([5.0, 2.0, 3.0, 0])
    insulator_length = np.array([0, 1.0, 1.0, 0])
    x_in = np.array(
        [
            [-99.83421563938445, -149.58689089845706, -166.41631740898708],
            [-88.72310452827334, -127.36466867623483, -133.08298407565374],
            [-77.61199341716224, -105.14244645401261, -99.74965074232041],
            [-66.50088230605112, -82.9202242317904, -66.41631740898708],
            [-55.38977119494001, -60.69800200956817, -33.08298407565374],
            [-44.278660083828896, -38.47577978734594, 0.25034925767960203],
            [-33.167548972717796, -16.253557565123742, 33.583682591012916],
            [-22.056437861606682, 5.968664657098486, 66.91701592434626],
            [-10.945326750495568, 28.190886879320715, 100.2503492576796],
            [0.1657843606155467, 50.41310910154294, 133.58368259101292],
        ]
    )
    y_in = np.zeros_like(x_in)
    z_in = np.array(
        [
            [10.000027484454499, 22.543635341861325, 27.950996373999338],
            [7.892465990400721, 16.309663919347848, 17.81588867365036],
            [6.035725912277301, 11.09573132895414, 9.98303742108253],
            [4.428890303670219, 6.891536749134786, 4.4176170485698],
            [3.0711656328586967, 3.6887742309041904, 1.0948831919131985],
            [1.9618813909314525, 1.4811162883232587, 6.267475205490314e-05],
            [1.1004897606577968, 0.2642013977043334, 1.1282878256807516],
            [0.4865653459493746, 0.03562538083679012, 4.484574836138955],
            [0.11980496177910194, 0.7949366552120196, 10.083846062855395],
            [2.7484454490078747e-05, 2.5436353418613056, 17.950996373999327],
        ]
    )

    x_out = np.array(
        [
            [0.0, 100.0, 300.0],
            [11.111111111111114, 122.22222222222223, 333.33333333333337],
            [22.222222222222214, 144.44444444444446, 366.6666666666667],
            [33.33333333333333, 166.66666666666666, 400.0],
            [44.44444444444444, 188.88888888888889, 433.33333333333337],
            [55.55555555555556, 211.11111111111111, 466.6666666666667],
            [66.66666666666666, 233.33333333333331, 500.0],
            [77.77777777777777, 255.55555555555554, 533.3333333333334],
            [88.88888888888889, 277.77777777777777, 566.6666666666667],
            [100.0, 300.0, 600.0],
        ]
    )
    y_out = np.array([[5.0, 2.0, 3.0]] * 10)
    z_out = np.array(
        [
            [48.0, 39.0, 19.0],
            [45.89243851, 32.76602858, 8.8648923],
            [44.03569843, 27.55209599, 1.03204105],
            [42.42886282, 23.34790141, -4.53337933],
            [41.07113815, 20.14513889, -7.85611318],
            [39.96185391, 17.93748095, -8.9509337],
            [39.10046228, 16.72056606, -7.82270855],
            [38.48653786, 16.49199004, -4.46642154],
            [38.11977748, 17.25130131, 1.13284969],
            [38.0, 19.0, 9.0],
        ]
    )

    displacement_vector = create_default_displacement_vector(insulator_length)
    x_1, y_1, z_1 = translate_cable_to_support(
        x_in,
        y_in,
        z_in,
        altitude,
        span_length,
        crossarm_length,
        insulator_length,
        line_angle=np.array([0, 0, 0, np.nan]),
        displacement_vector=displacement_vector,
        ground_altitude=np.array([0, 0, 0, 0]),
    )

    np.testing.assert_almost_equal(x_1, x_out)
    np.testing.assert_almost_equal(y_1, y_out)
    np.testing.assert_almost_equal(z_1, z_out)
