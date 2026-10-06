import numpy as np
import pytest

from mechaphlowers.core.tangential_sighting.tangential_sighting import (
    compute_parameter__array,
    compute_parameter__scalar,
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


def test_angle_to_cable_tangent_zero_raises() -> None:
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
    input_height or distance must be provided (not zero, not nan).

    Indeed, distance = 0 means the sighting device is directly below
    the left hanging point, in which case input_height is required.
    """
    with pytest.raises(
        ValueError,
        match="input_height .* distance",
    ):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([np.pi / 3]),
            angle_to_left_support=np.array([np.pi / 2]),
            angle_to_right_support=np.array([np.pi / 2]),
            span_length=np.array([500]),
            input_height=np.array([0.0]),
            distance=np.array([0.0]),
        )


def test_zero_angle_to_left_support_and_positive_distance_raises() -> None:
    # If the angle to the left support is zero, the distance must be zero or nan.
    with pytest.raises(
        ValueError,
        match="angle to the left support .* distance",
    ):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([np.pi / 3]),
            angle_to_left_support=np.array([0.0]),
            angle_to_right_support=np.array([np.pi / 2]),
            span_length=np.array([500]),
            input_height=np.array([0.0]),
            distance=np.array([20]),
        )


def test_zero_distance_and_positive_angle_to_left_support_raises() -> None:
    # If the distance is zero, the angle to the left support must be zero.
    with pytest.raises(
        ValueError,
        match="distance .* angle to the left support",
    ):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([1.6]),
            angle_to_left_support=np.array([0.2]),
            angle_to_right_support=np.array([1.5]),
            span_length=np.array([400]),
            input_height=np.array([20]),
            distance=np.array([0]),
        )


def test_nan_distance_and_positive_angle_to_left_support_raises() -> None:
    # If the distance is nan, the angle to the left support must be zero.
    with pytest.raises(
        ValueError,
        match="distance .* angle to the left support",
    ):
        compute_parameter__array(
            angle_to_cable_tangent=np.array([1.6]),
            angle_to_left_support=np.array([0.2]),
            angle_to_right_support=np.array([1.5]),
            span_length=np.array([400]),
            input_height=np.array([20]),
            distance=np.array([np.nan]),
        )


def test_input_height_nan_ok_if_distance_provided() -> None:
    result = compute_parameter__array(
        angle_to_cable_tangent=convert_grad_to_rad(np.array([95.622])),
        angle_to_left_support=convert_grad_to_rad(np.array([75.776])),
        angle_to_right_support=convert_grad_to_rad(np.array([94.228])),
        span_length=np.array([500]),
        input_height=np.array([np.nan]),
        distance=np.array([-50]),
    )
    np.testing.assert_allclose(
        result,
        2199.3,
        atol=1e-1,
    )


def test_distance_nan_ok_if_input_height_provided() -> None:
    result = compute_parameter__array(
        angle_to_cable_tangent=convert_grad_to_rad(np.array([98.999])),
        angle_to_left_support=convert_grad_to_rad(np.array([0.000])),
        angle_to_right_support=convert_grad_to_rad(np.array([96.820])),
        span_length=np.array([400]),
        input_height=np.array([20]),
        distance=np.array([np.nan]),
    )
    np.testing.assert_allclose(
        result,
        1200.0,
        atol=1e-1,
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
        (
            convert_grad_to_rad(np.array([101.607, 91.631])),
            convert_grad_to_rad(np.array([50, 91.562])),
            convert_grad_to_rad(np.array([100, 91.562])),
            np.array([800, 300]),
            np.array([0, 0]),
            np.array([50, 150]),
            np.array([2500.1, 1000.4]),
        ),
    ],
    ids=[
        "left of left support - 1st test case of prototype doc",
        "at support - 2nd test case of prototype doc",
        "between supports - 3rd test case of prototype doc",
        "between supports - 4th test case of prototype doc",
        "between supports - 3th and 4th test cases of prototype doc together",
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
    angle_to_cable_tangent = convert_grad_to_rad(np.array([91.631]))
    angle_to_left_support = convert_grad_to_rad(np.array([124.224]))
    angle_to_right_support = convert_grad_to_rad(np.array([106.345]))

    with pytest.raises(ConvergenceError):
        compute_parameter__array(
            angle_to_cable_tangent,
            angle_to_left_support,
            angle_to_right_support,
            np.array([250]),
            np.array([0]),
            np.array([-50]),
        )


def test_compute_parameter__scalar_ok() -> None:
    result = compute_parameter__scalar(
        convert_grad_to_rad(95.622),
        convert_grad_to_rad(75.776),
        convert_grad_to_rad(94.228),
        500,
        0,
        -50,
    )
    np.testing.assert_allclose(result, 2199.3, atol=1e-1)


def test_compute_parameter__scalar_non_convergent_case_raises() -> None:
    angle_to_cable_tangent = convert_grad_to_rad(91.631)
    angle_to_left_support = convert_grad_to_rad(124.224)
    angle_to_right_support = convert_grad_to_rad(106.345)

    with pytest.raises(ConvergenceError):
        compute_parameter__scalar(
            angle_to_cable_tangent,
            angle_to_left_support,
            angle_to_right_support,
            250,
            0,
            -50,
        )
