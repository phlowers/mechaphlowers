# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import sys

import numpy as np
import pytest

from mechaphlowers.core.geometry.distances import (
    DistanceEngine,
    DistanceResult,
    batch_plane_distances,
    get_projection_points,
    minimize_bounded,
    plane_basis,
    points_distance_inside_plane,
)
from mechaphlowers.core.geometry.planes import plane_from_line


@pytest.mark.parametrize("without_scipy", [False, True])
def test_minimize_bounded(monkeypatch, without_scipy):
    if without_scipy:
        monkeypatch.setitem(sys.modules, "scipy.optimize", None)

    x_min = minimize_bounded(lambda x: (x - 0.3) ** 2, (0.0, 1.0), xtol=1e-6)
    np.testing.assert_allclose(x_min, 0.3, atol=1e-5)

    # Minimum on the boundary
    x_min = minimize_bounded(lambda x: x, (0.2, 1.0), xtol=1e-6)
    np.testing.assert_allclose(x_min, 0.2, atol=1e-5)


def test_points_distance_inside_plane_returns_distance_and_projections():
    point_2 = np.array([1.0, 0.0, 0.0])
    point_1 = np.array([0.0, 0.0, 0.0])
    # plane_normal = np.array([0.0, 0.0, 1.0])
    # line_direction_normalized = np.array([1.0, 0.0, 0.0])
    u_plane = np.array([1.0, 0.0, 0.0])
    v_plane = np.array([0.0, 1.0, 0.0])

    distance_3d, projection_u, projection_v = points_distance_inside_plane(
        point_1,
        point_2,
        u_plane,
        v_plane,
    )

    np.testing.assert_allclose(distance_3d, 1.0)
    np.testing.assert_allclose(projection_u, 1.0)
    np.testing.assert_allclose(projection_v, 0.0)


def test_points_distance_inside_plane_invalid_projection_raises():
    point_2 = np.array([1.0, 0.0, 0.0])
    point_1 = np.array([0.0, 0.0, 0.0])
    # plane_normal = np.array([0.0, 0.0, 1.0])
    # line_direction_normalized = np.array([1.0, 0.0, 0.0])
    u_plane = np.array([0.0, 1.0, 0.0])
    v_plane = np.array([0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="Projected distance"):
        points_distance_inside_plane(
            point_1,
            point_2,
            u_plane,
            v_plane,
        )


def test_get_projection_points():
    obstacle = np.array([10.0, 20.0, 30.0])
    projection_u = 2.0
    projection_v = -3.0
    u_plane = np.array([1.0, 0.0, 0.0])
    v_plane = np.array([0.0, 0.0, 1.0])

    u_point, v_point = get_projection_points(
        obstacle, projection_u, projection_v, u_plane, v_plane
    )

    np.testing.assert_allclose(u_point, np.array([12.0, 20.0, 30.0]))
    np.testing.assert_allclose(v_point, np.array([10.0, 20.0, 27.0]))


def test_distance_result_reverse_y_axis():
    distance_result = DistanceResult(
        np.array([100.0000, 6.7000, 0.0000]),
        np.array([100.0000, 9.9940, 49.9035]),
        np.array([-0.0000, 1.0000, 0.0000]),
        np.array(
            [0.0000, 0.1000, 1.0000]
        ),  # incorrect value, just for testing reversed y
        50.894348,
        9.993974,
        49.903458,
    )
    reversed_distance_result = distance_result.generate_with_reversed_y_axis()
    # Check y axis is correctly reversed
    np.testing.assert_allclose(
        reversed_distance_result.point_base,
        np.array([100.0000, -6.7000, 0.0000]),
    )
    np.testing.assert_allclose(
        reversed_distance_result.point_target,
        np.array([100.0000, -9.9940, 49.9035]),
    )
    np.testing.assert_allclose(
        reversed_distance_result.u_plane, np.array([-0.0000, -1.0000, 0.0000])
    )
    np.testing.assert_allclose(
        reversed_distance_result.v_plane, np.array([0.0000, -0.1000, 1.0000])
    )

    # Check y axis is NOT reversed on original object
    np.testing.assert_allclose(
        distance_result.point_base, np.array([100.0000, 6.7000, 0.0000])
    )
    np.testing.assert_allclose(
        distance_result.point_target, np.array([100.0000, 9.9940, 49.9035])
    )
    np.testing.assert_allclose(
        distance_result.u_plane, np.array([-0.0000, 1.0000, 0.0000])
    )
    np.testing.assert_allclose(
        distance_result.v_plane, np.array([0.0000, 0.1000, 1.0000])
    )


def test_plane_distances_matches_scalar_plane_distance():
    engine = DistanceEngine()
    engine.add_span_frame(
        np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0])
    )
    x = np.linspace(0, 100, 30)
    curve = np.column_stack([x, np.zeros_like(x), 50 - 0.002 * (x - 50) ** 2])
    engine.add_curves(curve)
    points = np.array(
        [
            [30.0, 2.0, 10.0],
            [60.0, -3.0, 20.0],
            [80.0, 1.0, 5.0],
            [10.0, 0.5, 30.0],
            [150.0, 0.0, 0.0],
        ]
    )

    distances, targets = engine.plane_distances(points)

    assert distances.shape == (5,)
    assert targets.shape == (5, 3)
    for i in range(4):
        expected = engine.plane_distance(points[i], frame="section")
        np.testing.assert_allclose(distances[i], expected.distance_3d)
        np.testing.assert_allclose(targets[i], expected.point_target)
    assert np.isinf(distances[4])
    assert np.all(np.isnan(targets[4]))


def test_plane_distances_accepts_single_point_and_keeps_state():
    engine = DistanceEngine()
    engine.add_span_frame(
        np.array([0.0, 0.0, 0.0]), np.array([100.0, 0.0, 0.0])
    )
    engine.add_curves(
        np.column_stack(
            [np.linspace(0, 100, 10), np.zeros(10), np.linspace(20, 10, 10)]
        )
    )
    engine.plane_distance(np.array([40.0, 0.0, 5.0]), frame="section")
    point_base_before = engine.point_base.copy()

    distances, targets = engine.plane_distances(np.array([40.0, 1.0, 5.0]))

    assert distances.shape == (1,)
    assert targets.shape == (1, 3)
    np.testing.assert_array_equal(engine.point_base, point_base_before)


def _two_spans_curves_and_normals():
    x = np.linspace(0.0, 100.0, 11)
    curve_0 = np.column_stack([x, 0.2 * x, -0.01 * x * (100.0 - x)])
    curve_1 = np.column_stack(
        [x, -0.5 * x + 10.0, -0.02 * x * (100.0 - x) + 3]
    )
    curves = np.stack([curve_0, curve_1])
    normals = np.array([[0.0, 1.0, 0.0], [0.6, 0.8, 0.0]])
    return curves, normals


def test_batch_plane_distances_matches_plane_distance_per_point():
    curves, normals = _two_spans_curves_and_normals()
    points = np.array(
        [
            [30.0, 5.0, -10.0],
            [40.0, -8.0, -6.0],
            [20.0, -2.0, -3.0],
            [40.0, 12.0, -8.0],
            1000.0 * normals[1],  # no intersection
        ]
    )
    span_indices = np.array([0, 1, 1, 0, 1])

    distances, targets = batch_plane_distances(
        points, span_indices, curves, normals
    )

    for i, (point, span) in enumerate(zip(points, span_indices)):
        engine = DistanceEngine()
        engine.add_span_frame(np.array([0.0, 0.0, 0.0]), normals[span] * 100.0)
        engine.add_curves(curves[span])
        if i == 4:
            assert np.isinf(distances[i])
            assert np.all(np.isnan(targets[i]))
            continue
        expected = engine.plane_distance(point, frame="section")
        np.testing.assert_allclose(
            distances[i], expected.distance_3d, atol=1e-9
        )
        np.testing.assert_allclose(
            targets[i], expected.point_target, atol=1e-9
        )


def test_plane_basis_matches_plane_from_line():
    normals = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],  # vertical
        ]
    )
    u, v = plane_basis(normals)

    assert u.shape == (4, 3)
    assert v.shape == (4, 3)
    for i, normal in enumerate(normals):
        u_expected, v_expected, _ = plane_from_line(np.zeros(3), normal)
        np.testing.assert_allclose(u[i], u_expected, atol=1e-12)
        np.testing.assert_allclose(v[i], v_expected, atol=1e-12)
