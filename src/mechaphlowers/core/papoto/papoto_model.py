# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np

try:
    from scipy import optimize  # type: ignore
except ImportError:
    import mechaphlowers.numeric.scipy as optimize

_ZETA = 1.0


def papoto_validity(
    parameter_1_2: np.ndarray,
    parameter_2_3: np.ndarray,
    parameter_1_3: np.ndarray,
) -> np.ndarray:
    """papoto_validity

    Function providing a validity criteria for the papoto method, based on the 3 computed values.

    Args:
        parameter_1_2 (np.float): first computed parameters
        parameter_2_3 (np.float): second computed parameters
        parameter_1_3 (np.float): third computed parameters

    Returns:
        np.float: validity criteria vector
    """

    validity = (
        np.diff(
            np.array(
                [
                    parameter_1_2,
                    parameter_2_3,
                    parameter_1_3,
                    parameter_1_2,
                ]
            ).T
        )
        / np.array([parameter_2_3, parameter_1_3, parameter_1_2]).T
    )

    return np.max(validity, axis=1)


def papoto_3_points(
    a: np.ndarray,
    HL: np.ndarray,
    VL: np.ndarray,
    HR: np.ndarray,
    VR: np.ndarray,
    H1: np.ndarray,
    V1: np.ndarray,
    H2: np.ndarray,
    V2: np.ndarray,
    H3: np.ndarray,
    V3: np.ndarray,
) -> np.ndarray:
    """Computes PAPOTO 3 times, and return the mean between those 3 values.

    Args:
        a (np.ndarray): Length of the span
        HL (np.ndarray): horizontal distance of the left part of the span
        VL (np.ndarray): vertical distance of the left part of the span
        HR (np.ndarray): horizontal distance of the right part of the span
        VR (np.ndarray): vertical distance of the right part of the span
        H1 (np.ndarray): horizontal distance of point 1
        V1 (np.ndarray): vertical distance of point 1
        H2 (np.ndarray): horizontal distance of point 2
        V2 (np.ndarray): vertical distance of point 2
        H3 (np.ndarray): horizontal distance of point 3
        V3 (np.ndarray): vertical distance of point 3
    Returns:
        parameter_mean (np.ndarray): mean of the 3 computed parameters
    """
    parameter_1_2 = papoto_2_points(a, HL, VL, HR, VR, H1, V1, H2, V2)
    parameter_2_3 = papoto_2_points(a, HL, VL, HR, VR, H2, V2, H3, V3)
    parameter_1_3 = papoto_2_points(a, HL, VL, HR, VR, H1, V1, H3, V3)
    parameter_mean = np.mean(
        np.array([parameter_1_2, parameter_2_3, parameter_1_3]), axis=0
    )
    return parameter_mean


def papoto_2_points(
    a: np.ndarray,
    HL: np.ndarray,
    VL: np.ndarray,
    HR: np.ndarray,
    VR: np.ndarray,
    H1: np.ndarray,
    V1: np.ndarray,
    H2: np.ndarray,
    V2: np.ndarray,
) -> np.ndarray:
    """Computes PAPOTO method with 2 points.

    Args:
        a (np.ndarray): Length of the span
        HL (np.ndarray): horizontal angle in rad of the left part of the span
        VL (np.ndarray): vertical angle in rad of the left part of the span
        HR (np.ndarray): horizontal angle in rad of the right part of the span
        VR (np.ndarray): vertical angle in rad of the right part of the span
        H1 (np.ndarray): horizontal angle in rad of point 1
        V1 (np.ndarray): vertical angle in rad of point 1
        H2 (np.ndarray): horizontal angle in rad of point 2
        V2 (np.ndarray): vertical angle in rad of point 2
    Returns:
        parameter (np.ndarray): parameter value
    """
    Alpha = HR - HL
    Alpha1 = H1 - HL
    Alpha2 = H2 - HL
    VL = np.pi / 2 - VL  # null angle = horizon
    VR = np.pi / 2 - VR
    V1 = np.pi / 2 - V1
    V2 = np.pi / 2 - V2

    nb_loops = 100

    iteration = (np.pi - Alpha) / 2
    AlphaR = np.zeros_like(Alpha)

    for loop_index in range(1, nb_loops):
        AlphaR = (
            AlphaR + iteration
        )  # for first loop: alphaD = (np.pi - alpha) / 2
        AlphaL = (
            np.pi - Alpha - AlphaR
        )  # for first loop: alphaG = (np.pi - alpha) / 2

        # computing distances between station and supports G and D
        distL = a / np.sin(Alpha) * np.sin(AlphaR)
        distR = distL * np.cos(Alpha) + a * np.cos(AlphaR)

        # computing attachment altitudes + elevation difference
        zL = distL * np.tan(VL)
        zR = distR * np.tan(VR)
        h = zR - zL

        # computing distances between station and points 1 and 2 + a1, a2, z1, z2
        dist1 = distL * np.sin(AlphaL) / np.sin(np.pi - Alpha1 - AlphaL)
        dist2 = distL * np.sin(AlphaL) / np.sin(np.pi - Alpha2 - AlphaL)

        a1 = distL * np.cos(AlphaL) + dist1 * np.cos(np.pi - Alpha1 - AlphaL)
        a2 = distL * np.cos(AlphaL) + dist2 * np.cos(np.pi - Alpha2 - AlphaL)

        z1 = dist1 * np.tan(V1)
        z2 = dist2 * np.tan(V2)

        # first approximation of parameter using parabola model
        p0 = a1 * (a - a1) / (2 * ((zL - z1) + h * a1 / a))
        p = parameter_solver(a, h, zL - z1, a1, p0)

        # computing an elevation difference using newly found parameter, and comparing with zG - z2
        # val: distance between lowest point with left support
        val = a / 2 - p * np.asinh(h / (2 * p * np.sinh(a / (2 * p))))
        dif = p * (np.cosh(val / p) - np.cosh((a2 - val) / p))

        iteration = (
            np.sign(dif - (zL - z2))
            * (np.pi - Alpha)
            / (2 ** (1 + loop_index))
        )

        stop_variable = abs(dif - (zL - z2))
        stop_value = 0.001
        if (
            np.logical_or(stop_variable < stop_value, np.isnan(stop_variable))
        ).all():
            break

    return p


def parameter_solver(
    a: np.ndarray,
    h: np.ndarray,
    delta: np.ndarray,
    x: np.ndarray,
    p0: np.ndarray,
    solver: str = "newton",
) -> np.ndarray:
    solver_dict = {"newton": optimize.newton}
    try:
        solver_method = solver_dict[solver]
    except KeyError:
        raise ValueError(f"Incorrect solver name: {solver}")

    solver_result = solver_method(
        function_f,
        p0,
        fprime=function_f_prime,
        args=(a, h, delta, x),
        maxiter=10,
        tol=1e-5,
        full_output=True,
    )
    if not solver_result.converged.all():
        raise ValueError("Solver did not converge")
    return solver_result.root


def function_f(
    p: np.ndarray,
    a: np.ndarray,
    h: np.ndarray,
    delta: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Function for which we want to find the root.

    $f(p) = p * (\\cosh(\\frac{val}{p}) - \\cosh(\\frac{x-val}{p}) - \\delta$

    with $val = \\frac{a}{2} - p * \\sinh^{-1}(\\frac{h} {2 * p * sinh(\\frac{a} {2 * p})})$

    Args:
        p (np.ndarray): parameter (variable to find)
        a (np.ndarray): span length
        h (np.ndarray): altitude difference between supports
        delta (np.ndarray): altitude difference between point 1 and left support
        x (np.ndarray): abscissa of chosen point 1

    Returns:
        np.ndarray: value of the function f at p
    """
    # val: distance between lowest point with left support
    val = a / 2 - p * np.asinh(h / (2 * p * np.sinh(a / 2 / p)))
    f = p * (np.cosh(val / p) - np.cosh((x - val) / p)) - delta
    return f


def function_f_prime(
    p: np.ndarray,
    a: np.ndarray,
    h: np.ndarray,
    delta: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Approximation of the derivate of function_f with respect to p, computed using finite difference.

    Args:
        p (np.ndarray): parameter (variable to find)
        a (np.ndarray): span length
        h (np.ndarray): altitude difference between supports
        delta (np.ndarray): altitude difference between point 1 and left support
        x (np.ndarray): abscissa of chosen point 1

    Returns:
        np.ndarray: approximation of $f'(p)$
    """
    return (
        function_f(p + _ZETA, a, h, delta, x) - function_f(p, a, h, delta, x)
    ) / _ZETA


def convert_grad_to_rad(angle_in_grad: np.ndarray) -> np.ndarray:
    """Converts an angle in grad to radians.

    Args:
        angle_in_grad (np.ndarray): array of angles in grad

    Returns:
        np.ndarray: array of angles in radians
    """
    return angle_in_grad / 200 * np.pi
