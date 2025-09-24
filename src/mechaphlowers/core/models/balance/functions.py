# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np


def L(p, x_n, x_m):
    return p * (np.sinh(x_n / p) - np.sinh(x_m / p))


def x_m(a, b, p):
    return -a / 2 + p * np.arcsinh(b / (2 * p * np.sinh(a / (2 * p))))


def x_n(a, b, p):
    return x_m(a, b, p) + a


def z(x, p, x_m):
    # repeating value to perform multidim operation
    xx = x.T + x_m  # >>> why + x_m?
    # self.p is a vector of size (nb support, ). I need to convert it in a matrix (nb support, 1) to perform matrix operation after.
    # Ex: self.p = array([20,20,20,20]) -> self.p([:,new_axis]) = array([[20],[20],[20],[20]])
    # pp = p[:, np.newaxis]

    rr = p * (np.cosh(xx / p) - 1)

    # reshaping back to p,x -> (vertical, horizontal)
    return rr.T


def T_moy(p, L, x_n, x_m, lineic_weight, k_load=None):
    a = x_n - x_m
    if k_load is None:
        k_load = np.ones_like(p)
    return (
        p
        * k_load
        * lineic_weight
        * (a + (np.sinh(2 * x_n / p) - np.sinh(2 * x_m / p)) * p / 2)
        / L
        / 2
    )


# def z_from_x_2ddl(span):

#     span.compute()
#     span
#     return b * np.sinh(x / p) / np.sinh(a / (2 * p))


def grad_to_rad(angles_grad: np.ndarray):
    return angles_grad * np.pi / 200


def grad_to_deg(angles_grad: np.ndarray):
    return angles_grad * 180 / 200


def deg_to_rad(angles_deg: np.ndarray):
    return angles_deg * np.pi / 180
