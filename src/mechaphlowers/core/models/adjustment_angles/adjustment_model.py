# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. if a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from typing import Literal, overload

import numpy as np


@overload
def compute_adjustment_angles(
    a: float,
    HL: float,
    VL: float,
    HR: float,
    VR: float,
    dist_support: float,
    parameter: float,
    side: Literal["right", "left"] = "left",
) -> tuple[float, float]: ...


@overload
def compute_adjustment_angles(
    a: np.ndarray,
    HL: np.ndarray,
    VL: np.ndarray,
    HR: np.ndarray,
    VR: np.ndarray,
    dist_support: np.ndarray,
    parameter: np.ndarray,
    side: Literal["right", "left"] = "left",
) -> tuple[np.ndarray, np.ndarray]: ...


def compute_adjustment_angles(
    a: float | np.ndarray,
    HL: float | np.ndarray,
    VL: float | np.ndarray,
    HR: float | np.ndarray,
    VR: float | np.ndarray,
    dist_support: float | np.ndarray,
    parameter: float | np.ndarray,
    side: Literal["right", "left"] = "left",
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Computation of the horizontal and vertical angles of the pointing station for adjustment cables.

    The station is at `dist_support` of the left or right support. Which support is specified with `side` argument

    All angles are in rad, distances are in meters

    Args:
        a (float | np.ndarray): length of the span
        HL (float | np.ndarray): horizontal angle with left support, in rad
        VL (float | np.ndarray): vertical angle with left support, in rad
        HR (float | np.ndarray): horizontal angle with right support, in rad
        VR (float | np.ndarray): vertical angle with right support, in rad
        dist_support (float | np.ndarray): distance between the station and the studied support, in meters
        parameter (float | np.ndarray): wanted sagging parameter
        side (Literal["right", "left"], optional): specify which support dist_support is refering to. Defaults to "left".

    Returns:
        tuple[float | np.ndarray, float | np.ndarray]: horizontal angle, vertical angle, in rad
    """

    VL = np.pi / 2 - VL
    VR = np.pi / 2 - VR

    alpha = HR - HL

    alpha_opposite_side = np.arcsin(np.sin(alpha) * dist_support / a)
    alpha_same_side = np.pi - alpha - alpha_opposite_side
    dist_opposite_support = a / np.sin(alpha) * np.sin(alpha_same_side)

    expected_alpha_opposite_value = np.arccos(
        -(dist_support**2 - dist_opposite_support**2 - a**2)
        / 2
        / dist_opposite_support
        / a
    )
    if not np.allclose(
        alpha_opposite_side,
        expected_alpha_opposite_value,
        atol=1e-6,
    ):
        alpha_opposite_side = expected_alpha_opposite_value
        alpha_same_side = np.pi - alpha - alpha_opposite_side
        dist_opposite_support = a / np.sin(alpha) * np.sin(alpha_same_side)

    if side == "left":
        alpha_left, alpha_right = alpha_same_side, alpha_opposite_side
        dist_left, dist_right = dist_support, dist_opposite_support
    elif side == "right":
        alpha_right, alpha_left = alpha_same_side, alpha_opposite_side
        dist_right, dist_left = dist_support, dist_opposite_support

    d = (
        (a / 2) ** 2
        + dist_left**2
        - 2 * dist_left * (a / 2) * np.cos(alpha_left)
    ) ** 0.5
    h = HR - np.arcsin(np.sin(alpha_right) / d * (a / 2))

    expected_h_value = HR - np.arccos(
        -((a / 2) ** 2 - dist_right**2 - d**2) / d / dist_right / 2
    )
    if not np.allclose(
        h,
        expected_h_value,
        atol=1e-6,
    ):
        h = expected_h_value

    # ' *************************************************
    # ' test bon fonctionnement
    # ' si le if précédent est positif, au final H = HHH

    # HHH = HL + np.arcsin(np.sin(alpha_left) / d * (a / 2))

    deniv = np.tan(VR) * dist_right - np.tan(VL) * dist_left

    x1 = -a / 2 + parameter * np.arcsinh(
        (deniv) / (2 * parameter * np.sinh(a / 2 / parameter))
    )
    z1 = parameter * (np.cosh(x1 / parameter) - 1)
    x = a / 2 + x1
    z = parameter * (np.cosh(x / parameter) - 1)

    V = np.arctan((dist_left * np.tan(VL) - z1 + z) / d)

    result_vertical_angle = np.pi / 2 - V

    return h, result_vertical_angle
