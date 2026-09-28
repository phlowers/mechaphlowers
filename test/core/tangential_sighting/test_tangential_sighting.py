import numpy as np
import pytest

from mechaphlowers.core.tangential_sighting.tangential_sighting import (
    compute_parameter__array,
)
from mechaphlowers.data.units import convert_grad_to_rad
from mechaphlowers.entities.errors import ConvergenceError

# Test input validation


def test_negative_angle_to_cable_tangent_raises() -> None:
    """angle_to_cable_tangent must not be negative."""
    with pytest.raises(ValueError, match="angle_to_cable_tangent"):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([-0.000001]),
            angle_to_left_support=np.array([np.pi / 2]),
            angle_to_right_support=np.array([np.pi / 2]),
            span_length=np.array([500]),
            input_height=np.array([0.0]),
            distance=np.array([100]),
        )


def test_angle_to_cable_tangent_zero_raises_value_error() -> None:
    with pytest.raises(ValueError, match="angle_to_cable_tangent == 0"):
        compute_parameter__array(
            np.array([0.0]),
            np.array([np.pi / 4]),
            np.array([np.pi / 4]),
            np.array([500.0]),
            np.array([0.0]),
            np.array([250.0]),
        )


def test_angle_to_cable_tangent_greater_than_pi_raises() -> None:
    """angle_to_cable_tangent must not be greater than pi."""
    with pytest.raises(ValueError, match="angle_to_cable_tangent"):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([2 * np.pi + 0.000001]),
            angle_to_left_support=np.array([np.pi / 2]),
            angle_to_right_support=np.array([np.pi / 2]),
            span_length=np.array([500]),
            input_height=np.array([0.0]),
            distance=np.array([100]),
        )


def test_negative_angle_to_left_support_raises() -> None:
    """angle_to_left_support must not be negative."""
    with pytest.raises(ValueError, match="angle_to_left_support"):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([np.pi / 3]),
            angle_to_left_support=np.array([-0.000001]),
            angle_to_right_support=np.array([np.pi / 2]),
            span_length=np.array([500]),
            input_height=np.array([0.0]),
            distance=np.array([100]),
        )


def test_angle_to_left_support_greater_than_pi_raises() -> None:
    """angle_to_left_support must not be greater than pi."""
    with pytest.raises(ValueError, match="angle_to_left_support"):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([np.pi / 3]),
            angle_to_left_support=np.array([2 * np.pi + 0.000001]),
            angle_to_right_support=np.array([np.pi / 2]),
            span_length=np.array([500]),
            input_height=np.array([0.0]),
            distance=np.array([100]),
        )


def test_angle_to_right_support_negative_raises() -> None:
    """angle_to_right_support must not be negative."""
    with pytest.raises(ValueError, match="angle_to_right_support"):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([np.pi / 3]),
            angle_to_left_support=np.array([np.pi / 2]),
            angle_to_right_support=np.array([-0.000001]),
            span_length=np.array([500]),
            input_height=np.array([0.0]),
            distance=np.array([100]),
        )


def test_angle_to_right_support_greater_than_pi_raises() -> None:
    """angle_to_right_support must not be greater than pi."""
    with pytest.raises(ValueError, match="angle_to_right_support"):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([np.pi / 3]),
            angle_to_left_support=np.array([np.pi / 2]),
            angle_to_right_support=np.array([2 * np.pi + np.pi / 2]),
            span_length=np.array([500]),
            input_height=np.array([0.0]),
            distance=np.array([100]),
        )


def test_span_length_zero_raises() -> None:
    """span_length must be strictly positive."""
    with pytest.raises(ValueError, match="span_length"):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([np.pi / 3]),
            angle_to_left_support=np.array([np.pi / 2]),
            angle_to_right_support=np.array([np.pi / 2]),
            span_length=np.array([0.0]),
            input_height=np.array([0.0]),
            distance=np.array([100]),
        )


def test_negative_span_length_raises() -> None:
    """span_length must be strictly positive."""
    with pytest.raises(ValueError, match="span_length"):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([np.pi / 3]),
            angle_to_left_support=np.array([np.pi / 2]),
            angle_to_right_support=np.array([np.pi / 2]),
            span_length=np.array([-0.000001]),
            input_height=np.array([0.0]),
            distance=np.array([100]),
        )


def test_negative_input_height_raises() -> None:
    """input_height must be positive."""
    with pytest.raises(ValueError, match="input_height"):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([np.pi / 3]),
            angle_to_left_support=np.array([np.pi / 2]),
            angle_to_right_support=np.array([np.pi / 2]),
            span_length=np.array([500]),
            input_height=np.array([-0.000001]),
            distance=np.array([100]),
        )


def test_input_height_and_distance_both_zero_raises() -> None:
    """
    input_height and distance can't both be zero.

    Indeed, distance = 0 means the sighting device is directly under (or in some
    rare cases above) the left hanging point, in which case input_height is
    required.
    """
    with pytest.raises(
        ValueError, match="input_height and distance can't both be zero"
    ):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([np.pi / 3]),
            angle_to_left_support=np.array([np.pi / 2]),
            angle_to_right_support=np.array([np.pi / 2]),
            span_length=np.array([500]),
            input_height=np.array([0.0]),
            distance=np.array([0.0]),
        )


def test_distance_nonzero_and_angle_to_left_support_zero_raises() -> None:
    """angle_to_left_support can't be zero if the distance is strictly positive."""
    with pytest.raises(
        ValueError,
        match="angle to left support can't be zero if distance isn't zero",
    ):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([np.pi / 3]),
            angle_to_left_support=np.array([0.0]),
            angle_to_right_support=np.array([np.pi / 2]),
            span_length=np.array([500]),
            input_height=np.array([0.0]),
            distance=np.array([20]),
        )


# Test passing cases: check results with results from prototype
@pytest.mark.parametrize(
    "angle_to_cable_tangent, angle_to_left_support, angle_to_right_support, "
    "span_length, input_height, distance, expected_result",
    [
        (
            convert_grad_to_rad(np.array([95.622])),
            convert_grad_to_rad(np.array([75.776])),
            convert_grad_to_rad(np.array([94.228])),
            np.array([500]),
            np.array([0]),
            np.array([-50]),
            np.array([2199.3]),
        ),
        (
            convert_grad_to_rad(np.array([98.999])),
            convert_grad_to_rad(np.array([0.000])),
            convert_grad_to_rad(np.array([96.820])),
            np.array([400]),
            np.array([20]),
            np.array([0]),
            np.array([1200.0]),
        ),
        (
            convert_grad_to_rad(np.array([101.607])),
            convert_grad_to_rad(np.array([50])),
            convert_grad_to_rad(np.array([100])),
            np.array([800]),
            np.array([0]),
            np.array([50]),
            np.array([2500.1]),
        ),
        (
            convert_grad_to_rad(np.array([91.631])),
            convert_grad_to_rad(np.array([91.562])),
            convert_grad_to_rad(np.array([91.562])),
            np.array([300]),
            np.array([0]),
            np.array([150]),
            np.array([1000.4]),
        ),
    ],
    ids=[
        "left of left support",
        "at support",
        "between supports-1",
        "between supports-2",
    ],
)
def test_compute_parameter_ok(
    angle_to_cable_tangent,
    angle_to_left_support,
    angle_to_right_support,
    span_length,
    input_height,
    distance,
    expected_result,
) -> None:
    result = compute_parameter__array(
        angle_to_cable_tangent,
        angle_to_left_support,
        angle_to_right_support,
        span_length,
        input_height,
        distance,
    )
    np.testing.assert_allclose(
        result,
        expected_result,
        atol=1e-1,
    )


def test_non_convergent_case_raises() -> None:
    """Test case adapted from tests cases provided by H. Ducloux"""
    with pytest.raises((RuntimeError, ConvergenceError)):
        compute_parameter__array(
            convert_grad_to_rad(np.array([91.631])),
            convert_grad_to_rad(np.array([124.224])),
            convert_grad_to_rad(np.array([106.345])),
            np.array([250]),
            np.array([0]),
            np.array([-50]),
        )
