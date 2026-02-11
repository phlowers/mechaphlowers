# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from mechaphlowers.core.geometry.planes import (
    intersection_curve_plane,
    line_function_from_2_points,
    meshgrid_plane,
    parametric_line_from_2_points,
    plane_from_line,
)


def test_parametric_line_from_2_points():
    p1 = np.array([1.0, 2.0, 3.0])
    p2 = np.array([4.0, 6.0, 3.0])

    origin, direction, direction_normalized = parametric_line_from_2_points(
        p1, p2
    )

    np.testing.assert_allclose(origin, p1)
    np.testing.assert_allclose(direction, p2 - p1)
    np.testing.assert_allclose(
        direction_normalized, direction / np.linalg.norm(direction)
    )
    np.testing.assert_allclose(np.linalg.norm(direction_normalized), 1.0)


def test_line_function_from_2_points_returns_points_on_line():
    p1 = np.array([-1.0, 0.0, 2.0])
    p2 = np.array([3.0, 4.0, -2.0])
    direction = p2 - p1

    line_fn = line_function_from_2_points(p1, p2)

    np.testing.assert_allclose(line_fn(0.0), p1)
    np.testing.assert_allclose(line_fn(1.0), p2)
    np.testing.assert_allclose(line_fn(0.25), p1 + 0.25 * direction)


def test_plane_from_line_z_normal():
    point = np.array([10.0, 20.0, 30.0])
    plane_normal = np.array([0.0, 0.0, 1.0])

    u_plane, v_plane, point_out = plane_from_line(point, plane_normal)

    np.testing.assert_allclose(point_out, point)
    np.testing.assert_allclose(u_plane, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(v_plane, np.array([0.0, 1.0, 0.0]))


def test_plane_from_line_orthonormal_basis():
    point = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([1.0, 2.0, 3.0])

    u_plane, v_plane, _ = plane_from_line(point, plane_normal)

    np.testing.assert_allclose(np.dot(u_plane, plane_normal), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.dot(v_plane, plane_normal), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.dot(u_plane, v_plane), 0.0, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(u_plane), 1.0, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(v_plane), 1.0, atol=1e-7)


def test_meshgrid_plane_shapes_and_center_point():
    u_plane = np.array([1.0, 0.0, 0.0])
    v_plane = np.array([0.0, 1.0, 0.0])
    point = np.array([10.0, 20.0, 30.0])

    x_plane, y_plane, z_plane = meshgrid_plane(
        u_plane,
        v_plane,
        point,
        scale_plane=2,
        grid_size_plane=3,
    )

    assert x_plane.shape == (3, 3)
    assert y_plane.shape == (3, 3)
    assert z_plane.shape == (3, 3)

    np.testing.assert_allclose(
        np.array([x_plane[1, 1], y_plane[1, 1], z_plane[1, 1]]), point
    )


def test_intersection_curve_plane_closest_points_when_not_fine_tuning():
    spans1 = np.array(
        [
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    plane_normal = np.array([1.0, 0.0, 0.0])
    point_on_plane = np.array([0.0, 0.0, 0.0])

    intersections = intersection_curve_plane(
        plane_normal, point_on_plane, spans1, fine_tuning=False
    )

    np.testing.assert_allclose(
        intersections, np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )


def test_intersection_curve_plane_fine_tuning_intersection():
    spans1 = np.array(
        [
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    plane_normal = np.array([1.0, 0.0, 0.0])
    point_on_plane = np.array([0.0, 0.0, 0.0])

    intersections = intersection_curve_plane(
        plane_normal, point_on_plane, spans1, fine_tuning=True
    )

    assert intersections.shape == (1, 3)
    np.testing.assert_allclose(intersections[0], np.array([0.0, 0.0, 0.0]))


def test_intersection_curve_plane_fine_tuning_same_side_raises():
    spans1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    plane_normal = np.array([1.0, 0.0, 0.0])
    point_on_plane = np.array([0.0, 0.0, 0.0])

    with pytest.raises(
        ValueError, match="Points are on the same side of the plane"
    ):
        intersection_curve_plane(
            plane_normal, point_on_plane, spans1, fine_tuning=True
        )
