# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.core.models.balance.models.utils_model_ducloux import VectorProjection


def test_proj_no_angles():
    # L_chain=np.array([3, 3, 3, 3]),
    # weight_chain=np.array([1000,  500, 500, 1000]),
    # arm_length=np.array([0,  0, 0, 0]),
    # line_angle=np.array([0, 0, 0, 0]),
    # x=np.array([0, 500, 800, 1200]),
    # z=np.array([30, 50, 60, 65]),
    # load=np.array([0, 0, 0, 0]),
    # cable=cable_AM600,

    Th = np.array([35316.0, 35316.0, 35316.0])
    Tv_g = np.array([-2974.3816, -1473.2471, -3064.9025])
    Tv_d = np.array([-5831.3273, -3832.06, -3957.3933])
    weight_chain = -np.array([1000.0, 500.0, 500.0, 1000.0])
    vector_projection = VectorProjection()
    vector_projection.set_all(
        Th=Th,
        Tv_g=Tv_g,
        Tv_d=Tv_d,
        alpha=np.array([0, 0, 0]),
        beta=np.array([0, 0, 0]),
        line_angle=np.array([0, 0, 0, 0]),
        proj_angle=np.array([0, 0, 0]),
        weight_chain=weight_chain,
    )
    T_attachment_left = vector_projection.T_attachments_plane_left()
    T_attachment_right = vector_projection.T_attachments_plane_right()
    T_line_left = vector_projection.T_line_plane_left()
    T_line_right = vector_projection.T_line_plane_right()
    Fx, Fy, Fz = vector_projection.forces()

    T_attachment_left_expected = np.array([Th, np.zeros_like(Th), Tv_g])
    T_attachment_right_expected = np.array([-Th, np.zeros_like(Th), Tv_d])
    T_line_left_expected = np.array([Th, np.zeros_like(Th), Tv_g])
    T_line_right_expected = np.array([-Th, np.zeros_like(Th), Tv_d])

    np.testing.assert_equal(T_attachment_left, T_attachment_left_expected)
    np.testing.assert_equal(T_attachment_right, T_attachment_right_expected)
    np.testing.assert_equal(T_line_left, T_line_left_expected)
    np.testing.assert_equal(T_line_right, T_line_right_expected)

    Fx_expected = np.array([Th[0], 0, 0, -Th[-1]])
    Fy_expected = np.zeros_like(Fx_expected)
    Fz_expected = np.array([-3474.3816, -7554.5744, -7146.9625, -4457.3933])

    np.testing.assert_equal(Fx, Fx_expected)
    np.testing.assert_equal(Fy, Fy_expected)
    np.testing.assert_allclose(Fz, Fz_expected)
