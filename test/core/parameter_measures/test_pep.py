# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest

from mechaphlowers.core.parameter_measures.pep import (
    _compute_parameter_from_one_point,
    compute_parameter,
)
from mechaphlowers.entities.errors import (
    ConvergenceError,
    MechaphlowersValueError,
)


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


def test_compute_parameter__x1_left_of_span__error() -> None:
    with pytest.raises(MechaphlowersValueError):
        compute_parameter(
            500,
            0,
            -10,
            -10,
            250,
            -5,
            300,
            -10,
        )


def test_compute_parameter__x1_right_of_span__error() -> None:
    with pytest.raises(MechaphlowersValueError):
        compute_parameter(
            500,
            0,
            500.1,
            -10,
            250,
            -5,
            300,
            -10,
        )


def test_compute_parameter__x1_too_close_to_left_support__warning() -> None:
    span_length = 500
    with pytest.raises(UserWarning):
        compute_parameter(
            span_length,
            elevation_difference=0,
            x1=0.09 * span_length,
            z1=-10,
            x2=250,
            z2=-5,
            x3=300,
            z3=-10,
        )


def test_compute_parameter__x1_too_close_to_right_support__warning() -> None:
    span_length = 500
    with pytest.raises(UserWarning):
        compute_parameter(
            span_length,
            elevation_difference=0,
            x1=0.91 * span_length,
            z1=-10,
            x2=250,
            z2=-5,
            x3=300,
            z3=-10,
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


def test_compute_parameter_from_one_point__measured_point_too_high_error() -> (
    None
):
    # It's physically impossible for the measured point to be higher than
    # both hanging points. (This would mean the parameter is negative and
    # the cable is floating in the air.)
    with pytest.raises(Exception):
        _compute_parameter_from_one_point(
            500,
            50,
            75,
            10,
        )


def test_compute_parameter_from_one_point__measured_point_near_hanging_point_error() -> (
    None
):
    # The measured point should be near the lowest point of the cable,
    # for the computed parameter to be precise.
    _compute_parameter_from_one_point(
        500,
        50,
        75,
        10,
    )
