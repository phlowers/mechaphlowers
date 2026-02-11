# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

from mechaphlowers.core.geometry.distances import (
    get_projection_points,
    points_distance_inside_plane,
)


def test_points_distance_inside_plane_returns_distance_and_projections():
    point_1 = np.array([1.0, 0.0, 0.0])
    point_2 = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([0.0, 0.0, 1.0])
    line_direction_normalized = np.array([1.0, 0.0, 0.0])
    u_plane = np.array([1.0, 0.0, 0.0])
    v_plane = np.array([0.0, 1.0, 0.0])

    distance_3d, projection_u, projection_v = points_distance_inside_plane(
        point_1,
        point_2,
        plane_normal,
        line_direction_normalized,
        u_plane,
        v_plane,
    )

    np.testing.assert_allclose(distance_3d, 1.0)
    np.testing.assert_allclose(projection_u, 1.0)
    np.testing.assert_allclose(projection_v, 0.0)


def test_points_distance_inside_plane_invalid_projection_raises():
    point_1 = np.array([1.0, 0.0, 0.0])
    point_2 = np.array([0.0, 0.0, 0.0])
    plane_normal = np.array([0.0, 0.0, 1.0])
    line_direction_normalized = np.array([1.0, 0.0, 0.0])
    u_plane = np.array([0.0, 1.0, 0.0])
    v_plane = np.array([0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="Projected distance"):
        points_distance_inside_plane(
            point_1,
            point_2,
            plane_normal,
            line_direction_normalized,
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
