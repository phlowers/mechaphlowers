# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. if a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from typing import Literal

import numpy as np


def compute_adjustment_angles(
    a,
    HG,
    VG,
    HD,
    VD,
    horizontal_distance_support,
    parameter,
    side: Literal["right", "left"] = "left",
):
    # Dim l, p, h, HG, HD, V, VG, VD, d, dG, DD, alpha, alphaG, alphaD, deniv, x1, z1, x, z As Double

    VG = np.pi / 2 - VG
    VD = np.pi / 2 - VD

    alpha = HD - HG

    if side == "left":
        dist_left = horizontal_distance_support
        alpha_right = np.arcsin(np.sin(alpha) * dist_left / a)
        alpha_left = np.pi - alpha - alpha_right
        dist_right = a / np.sin(alpha) * np.sin(alpha_left)

        if (
            round(
                alpha_right
                - np.arccos(
                    -(dist_left**2 - dist_right**2 - a**2) / 2 / dist_right / a
                ),
                6,
            )
            != 0
        ):
            alpha_right = np.arccos(
                -(dist_left**2 - dist_right**2 - a**2) / 2 / dist_right / a
            )
            alpha_left = np.pi - alpha - alpha_right
            dist_right = a / np.sin(alpha) * np.sin(alpha_left)

    # if DD != 0 And dG = 0:
    elif side == "right":
        dist_right = horizontal_distance_support
        alpha_left = np.arcsin(np.sin(alpha) * dist_right / a)
        alpha_right = np.pi - alpha - alpha_left
        dist_left = a / np.sin(alpha) * np.sin(alpha_right)

        if (
            round(
                alpha_left
                - np.arccos(
                    -(dist_right**2 - dist_left**2 - a**2) / 2 / dist_left / a
                ),
                6,
            )
            != 0
        ):
            alpha_left = np.arccos(
                -(dist_right**2 - dist_left**2 - a**2) / 2 / dist_left / a
            )
            alpha_right = np.pi - alpha - alpha_left
            dist_left = a / np.sin(alpha) * np.sin(alpha_right)

    d = (
        (a / 2) ** 2
        + dist_left**2
        - 2 * dist_left * (a / 2) * np.cos(alpha_left)
    ) ** 0.5
    h = HD - np.arcsin(np.sin(alpha_right) / d * (a / 2))

    if (
        round(
            (HD - h)
            - np.arccos(
                -((a / 2) ** 2 - dist_right**2 - d**2) / d / dist_right / 2
            ),
            6,
        )
        != 0
    ):
        h = HD - np.arccos(
            -((a / 2) ** 2 - dist_right**2 - d**2) / d / dist_right / 2
        )

    # ' *************************************************
    # ' test bon fonctionnement
    # ' si le if précédent est positif, au final H = HHH

    # HHH = HG + np.arcsin(np.sin(alpha_left) / d * (a / 2))

    result_horizontal_angle = h / np.pi * 200

    deniv = np.tan(VD) * dist_right - np.tan(VG) * dist_left

    x1 = -a / 2 + parameter * np.arcsinh(
        (deniv) / (2 * parameter * np.sinh(a / 2 / parameter))
    )
    z1 = parameter * (np.cosh(x1 / parameter) - 1)
    x = a / 2 + x1
    z = parameter * (np.cosh(x / parameter) - 1)

    V = np.arctan((dist_left * np.tan(VG) - z1 + z) / d)

    result_vertical_angle = 100 - V / np.pi * 200

    return result_horizontal_angle, result_vertical_angle
