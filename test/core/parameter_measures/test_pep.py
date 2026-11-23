# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest

from mechaphlowers.core.parameter_measures.pep import compute_parameter
from mechaphlowers.entities.errors import ConvergenceError


@pytest.mark.parametrize(
    "span_length,elevation_difference,x1,z1,x2,z2,x3,z3,"
    "expected_p1,expected_p2,expected_p3,expected_p",
    [
        (
            466.067,
            -1.560,
            198.332,
            -13.288,
            205.037,
            -13.444,
            298.816,
            -12.879,
            2105,
            2100,
            2106,
            2104,
        ),
        (
            485.144,
            7.023,
            123.346,
            -8.779,
            202.985,
            -10.631,
            250.684,
            -10.299,
            2115,
            2113,
            2113,
            2113,
        ),
        (
            430.521,
            -0.025,
            108.903,
            -8.270,
            221.108,
            -10.896,
            284.145,
            -9.803,
            2121,
            2129,
            2127,
            2126,
        ),
    ],
)
def test_compute_parameter_ok(
    span_length: float,
    elevation_difference: float,
    x1: float,
    z1: float,
    x2: float,
    z2: float,
    x3: float,
    z3: float,
    expected_p1: float,
    expected_p2: float,
    expected_p3: float,
    expected_p: float,
) -> None:
    p1, p2, p3, p, relative_error = compute_parameter(
        span_length,
        elevation_difference,
        x1,
        z1,
        x2,
        z2,
        x3,
        z3,
    )
    assert p1 == pytest.approx(expected_p1, abs=1)
    assert p2 == pytest.approx(expected_p2, abs=1)
    assert p3 == pytest.approx(expected_p3, abs=1)
    assert p == pytest.approx(expected_p, abs=1)
    assert relative_error < 0.05


def test_compute_parameter__raises_convergence_error() -> None:
    with pytest.raises(ConvergenceError):
        compute_parameter(
            500,
            0,
            200,
            -10,
            250,
            -5,
            300,
            -10,
        )


def test_compute_parameter__parameter_far_from_initial_guess() -> None:
    # Non-regression test added after finding out that compute_parameter
    # failed if the expected parameter was too small.
    # Indeed, the first implementation used the Newton resolution method
    # with an arbitrary, hard-coded initial guess, and it couldn't find the
    # right root if the parameter was too far away from the initial
    # guess.
    _, _, _, p, _ = compute_parameter(
        600,
        0,
        x1=270.00,
        z1=-91.83,
        x2=300.00,
        z2=-92.73,
        x3=330.00,
        z3=-91.83,
    )

    # then
    assert p == pytest.approx(500, abs=1)
