import numpy as np
import pytest

from mechaphlowers.core.tangential_sighting.tangential_sighting import (
    compute_parameter,
)

# Default valid inputs used as a baseline for each test.
# Each test overrides only the input(s) relevant to the case being checked.
_VALID_ANGLE_TO_CABLE_TANGENT = np.array([1.0])
_VALID_ANGLE_TO_LEFT_SUPPORT = np.array([1.0])
_VALID_ANGLE_TO_RIGHT_SUPPORT = np.array([1.0])
_VALID_SPAN_LENGTH = np.array([100.0])
_VALID_INPUT_HEIGHT = np.array([5.0])
_VALID_DISTANCE = np.array([2.0])


def _call_compute_parameter(
    angle_to_cable_tangent=None,
    angle_to_left_support=None,
    angle_to_right_support=None,
    span_length=None,
    input_height=None,
    distance=None,
):
    return compute_parameter(
        angle_to_cable_tangent
        if angle_to_cable_tangent is not None
        else _VALID_ANGLE_TO_CABLE_TANGENT.copy(),
        angle_to_left_support
        if angle_to_left_support is not None
        else _VALID_ANGLE_TO_LEFT_SUPPORT.copy(),
        angle_to_right_support
        if angle_to_right_support is not None
        else _VALID_ANGLE_TO_RIGHT_SUPPORT.copy(),
        span_length if span_length is not None else _VALID_SPAN_LENGTH.copy(),
        input_height
        if input_height is not None
        else _VALID_INPUT_HEIGHT.copy(),
        distance if distance is not None else _VALID_DISTANCE.copy(),
    )


def test_angle_to_cable_tangent_negative_raises() -> None:
    """angle_to_cable_tangent must not be lower than 0."""
    with pytest.raises(ValueError, match="angle_to_cable_tangent"):
        _call_compute_parameter(
            angle_to_cable_tangent=np.array([-0.000001]),
        )


def test_angle_to_cable_tangent_greater_than_pi_raises() -> None:
    """angle_to_cable_tangent must not be greater than pi."""
    with pytest.raises(ValueError, match="angle_to_cable_tangent"):
        _call_compute_parameter(
            angle_to_cable_tangent=np.array([np.pi + 0.000001]),
        )


def test_angle_to_left_support_negative_raises() -> None:
    """angle_to_left_support must not be lower than 0."""
    with pytest.raises(ValueError, match="angle_to_left_support"):
        _call_compute_parameter(
            angle_to_left_support=np.array([-0.000001]),
            distance=np.array([0.0]),
        )


def test_angle_to_left_support_greater_than_pi_raises() -> None:
    """angle_to_left_support must not be greater than pi."""
    with pytest.raises(ValueError, match="angle_to_left_support"):
        _call_compute_parameter(
            angle_to_left_support=np.array([np.pi + 0.000001]),
        )


def test_angle_to_right_support_negative_raises() -> None:
    """angle_to_right_support must not be lower than 0."""
    with pytest.raises(ValueError, match="angle_to_right_support"):
        _call_compute_parameter(
            angle_to_right_support=np.array([-0.000001]),
        )


def test_angle_to_right_support_greater_than_pi_raises() -> None:
    """angle_to_right_support must not be greater than pi."""
    with pytest.raises(ValueError, match="angle_to_right_support"):
        _call_compute_parameter(
            angle_to_right_support=np.array([np.pi + 0.000001]),
        )


def test_span_length_zero_raises() -> None:
    """span_length must be strictly positive."""
    with pytest.raises(ValueError, match="span_length"):
        _call_compute_parameter(
            span_length=np.array([0.0]),
        )


def test_span_length_negative_raises() -> None:
    """span_length must be strictly positive."""
    with pytest.raises(ValueError, match="span_length"):
        _call_compute_parameter(
            span_length=np.array([-0.000001]),
        )


def test_input_height_negative_raises() -> None:
    """input_height must not be negative."""
    with pytest.raises(ValueError, match="input_height"):
        _call_compute_parameter(
            input_height=np.array([-0.000001]),
        )


def test_input_height_and_distance_both_zero_raises() -> None:
    with pytest.raises(
        ValueError, match="input_height and distance can't both be zero"
    ):
        _call_compute_parameter(
            input_height=np.array([0.0]),
            distance=np.array([0.0]),
        )


def test_distance_nonzero_and_angle_to_left_support_zero_raises() -> None:
    """angle_to_left_support can't be zero if the distance isn't zero."""
    with pytest.raises(
        ValueError,
        match="angle to left support can't be zero if distance isn't zero",
    ):
        _call_compute_parameter(
            angle_to_left_support=np.array([0.0]),
            distance=np.array([2.0]),
        )


def test_compute_parameter_angle_to_cable_tangent_zero_raises_value_error() -> (
    None
):
    with pytest.raises(ValueError, match="angle_to_cable_tangent == 0"):
        compute_parameter(
            np.array([0.0]),
            np.array([np.pi / 4]),
            np.array([np.pi / 4]),
            np.array([500.0]),
            np.array([0.0]),
            np.array([250.0]),
        )


# Each case below is built by forward-simulating the catenary geometry for
# a known parameter (500.0) and a chosen (distance, elevation_difference)
# combination, so that the angle/height inputs are physically
# self-consistent and the true root is (approximately) known in advance.
# This covers the 12 combinations required by the issue:
#   distance in {0, > 0, < 0, == span_length}
#   x
#   elevation_difference in {0, positive, negative}
convergent_cases_inputs = [
    (
        np.array([1.4711276743037345]),
        np.array([1.0]),
        np.array([1.4376973019239723]),
        np.array([300.0]),
        np.array([40.167179953346135]),
        np.array([0.0]),
        np.array([500.0000000007603]),
    ),
    (
        np.array([1.4711276743037345]),
        np.array([1.0]),
        np.array([1.4210889737182058]),
        np.array([300.0]),
        np.array([30.16717995334614]),
        np.array([100.0]),
        np.array([500.0000000014593]),
    ),
    (
        np.array([1.4711276743037345]),
        np.array([1.0]),
        np.array([1.4424565472065516]),
        np.array([300.0]),
        np.array([45.167179953346135]),
        np.array([-50.0]),
        np.array([500.0000000006596]),
    ),
    (
        np.array([1.4711276743037345]),
        np.array([1.0]),
        np.array([1.4213830670732177]),
        np.array([300.0]),
        np.array([40.16053860683608]),
        np.array([0.0]),
        np.array([500.00000000522596]),
    ),
    (
        np.array([1.4711276743037345]),
        np.array([1.0]),
        np.array([1.3967719256116093]),
        np.array([300.0]),
        np.array([30.160538606836077]),
        np.array([100.0]),
        np.array([500.0000000085167]),
    ),
    (
        np.array([1.4711276743037345]),
        np.array([1.0]),
        np.array([1.4284497929853694]),
        np.array([300.0]),
        np.array([45.16053860683608]),
        np.array([-50.0]),
        np.array([500.00000000451513]),
    ),
    (
        np.array([1.4711276743037345]),
        np.array([1.0]),
        np.array([1.454083112346287]),
        np.array([300.0]),
        np.array([40.17382186343617]),
        np.array([0.0]),
        np.array([500.0000000005485]),
    ),
    (
        np.array([1.4711276743037345]),
        np.array([1.0]),
        np.array([1.4455856855785785]),
        np.array([300.0]),
        np.array([30.173821863436167]),
        np.array([100.0]),
        np.array([500.00000000367857]),
    ),
    (
        np.array([1.4711276743037345]),
        np.array([1.0]),
        np.array([1.4565141162748227]),
        np.array([300.0]),
        np.array([45.17382186343617]),
        np.array([-50.0]),
        np.array([500.0000000003441]),
    ),
]


# TODO
@pytest.mark.parametrize(
    "angle_to_cable_tangent, angle_to_left_support, angle_to_right_support, "
    "span_length, input_height, distance, expected",
    convergent_cases_inputs,
    ids=[
        "distance_zero-no_elevation_difference",
        "distance_positive-no_elevation_difference",
        "distance_negative-no_elevation_difference",
        "distance_zero-positive_elevation_difference",
        "distance_positive-positive_elevation_difference",
        "distance_negative-positive_elevation_difference",
        "distance_zero-negative_elevation_difference",
        "distance_positive-negative_elevation_difference",
        "distance_negative-negative_elevation_difference",
    ],
)
def test_compute_parameter_ok(
    angle_to_cable_tangent: np.ndarray,
    angle_to_left_support: np.ndarray,
    angle_to_right_support: np.ndarray,
    span_length: np.ndarray,
    input_height: np.ndarray,
    distance: np.ndarray,
    expected: np.ndarray,
) -> None:
    result = compute_parameter(
        angle_to_cable_tangent,
        angle_to_left_support,
        angle_to_right_support,
        span_length,
        input_height,
        distance,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-4)


_NON_CONVERGENT_CASES = [
    pytest.param(
        np.array([1.4]),
        np.array([1.4]),
        np.array([1.4]),
        np.array([100.0]),
        np.array([2.0]),
        np.array([2.0]),
        id="mismatched_nominal_geometry",
    ),
    pytest.param(
        np.array([1.4]),
        np.array([1.4]),
        np.array([1.4]),
        np.array([100.0]),
        np.array([2.0]),
        np.array([0.0]),
        id="mismatched_distance_zero",
    ),
    pytest.param(
        np.array([1.4]),
        np.array([1.4]),
        np.array([1.4]),
        np.array([1000.0]),
        np.array([5.0]),
        np.array([5.0]),
        id="mismatched_large_span_length",
    ),
    pytest.param(
        np.array([1.4]),
        np.array([1.4]),
        np.array([1.4]),
        np.array([10.0]),
        np.array([0.2]),
        np.array([0.5]),
        id="mismatched_small_span_length",
    ),
    pytest.param(
        np.array([0.01]),
        np.array([1.4]),
        np.array([1.4]),
        np.array([100.0]),
        np.array([5.0]),
        np.array([2.0]),
        id="angle_to_cable_tangent_near_zero",
    ),
    pytest.param(
        np.array([np.pi - 0.01]),
        np.array([1.4]),
        np.array([1.4]),
        np.array([100.0]),
        np.array([5.0]),
        np.array([2.0]),
        id="angle_to_cable_tangent_near_pi",
    ),
]


@pytest.mark.parametrize(
    "angle_to_cable_tangent, angle_to_left_support, angle_to_right_support, "
    "span_length, input_height, distance",
    _NON_CONVERGENT_CASES,
)
def test_compute_parameter_non_convergent_case_raises_runtime_error(
    angle_to_cable_tangent: np.ndarray,
    angle_to_left_support: np.ndarray,
    angle_to_right_support: np.ndarray,
    span_length: np.ndarray,
    input_height: np.ndarray,
    distance: np.ndarray,
) -> None:
    """Mismatched/poorly-conditioned inputs make the Newton solver itself
    fail to converge (zero derivative or max iterations reached); scipy
    raises RuntimeError directly, before _parameter_solver's own
    ConvergenceError check is ever reached.
    """
    with pytest.raises(RuntimeError, match="converge"):
        compute_parameter(
            angle_to_cable_tangent,
            angle_to_left_support,
            angle_to_right_support,
            span_length,
            input_height,
            distance,
        )
