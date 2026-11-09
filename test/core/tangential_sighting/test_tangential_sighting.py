import numpy as np
import pytest

from mechaphlowers.core.tangential_sighting.tangential_sighting import (
    compute_parameter__array,
    compute_parameter__scalar,
    compute_tangent_angle__array,
    compute_tangent_angle__scalar,
)
from mechaphlowers.data.units import convert_angle_to_rad
from mechaphlowers.entities.errors import (
    ConvergenceError,
    MechaphlowersValueError,
)

# Test input validation for compute_parameter__array


class TestComputeParameter:
    def test_negative_angle_to_cable_tangent_raises(self) -> None:
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

    def test_angle_to_cable_tangent_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="x must be different from 0"):
            compute_parameter__array(
                np.array([0.0]),
                np.array([np.pi / 4]),
                np.array([np.pi / 4]),
                np.array([500.0]),
                np.array([0.0]),
                np.array([250.0]),
            )

    def test_angle_to_cable_tangent_greater_than_pi_raises(self) -> None:
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

    def test_negative_angle_to_left_support_raises(self) -> None:
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

    def test_angle_to_left_support_greater_than_pi_raises(self) -> None:
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

    def test_angle_to_right_support_negative_raises(self) -> None:
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

    def test_angle_to_right_support_greater_than_pi_raises(self) -> None:
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

    def test_span_length_zero_raises(self) -> None:
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

    def test_negative_span_length_raises(self) -> None:
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

    def test_negative_input_height_raises(self) -> None:
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

    def test_input_height_and_distance_both_zero_raises(self) -> None:
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

    def test_zero_angle_to_left_support_and_positive_distance_raises(
        self,
    ) -> None:
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

    def test_zero_distance_and_positive_angle_to_left_support_raises(
        self,
    ) -> None:
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

    def test_nan_distance_and_positive_angle_to_left_support_raises(
        self,
    ) -> None:
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

    def test_input_height_nan_ok_if_distance_provided(self) -> None:
        result = compute_parameter__array(
            angle_to_cable_tangent=convert_angle_to_rad(np.array([95.622])),
            angle_to_left_support=convert_angle_to_rad(np.array([75.776])),
            angle_to_right_support=convert_angle_to_rad(np.array([94.228])),
            span_length=np.array([500]),
            input_height=np.array([np.nan]),
            distance=np.array([-50]),
        )
        np.testing.assert_allclose(
            result,
            2199.3,
            atol=1e-1,
        )

    def test_distance_nan_ok_if_input_height_provided(self) -> None:
        result = compute_parameter__array(
            angle_to_cable_tangent=convert_angle_to_rad(np.array([98.999])),
            angle_to_left_support=convert_angle_to_rad(np.array([0.000])),
            angle_to_right_support=convert_angle_to_rad(np.array([96.820])),
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
                convert_angle_to_rad(np.array([95.622])),
                convert_angle_to_rad(np.array([75.776])),
                convert_angle_to_rad(np.array([94.228])),
                np.array([500]),
                np.array([0]),
                np.array([-50]),
                np.array([2199.3]),
            ),
            (
                convert_angle_to_rad(np.array([98.999])),
                convert_angle_to_rad(np.array([0.000])),
                convert_angle_to_rad(np.array([96.820])),
                np.array([400]),
                np.array([20]),
                np.array([0]),
                np.array([1200.0]),
            ),
            (
                convert_angle_to_rad(np.array([101.607])),
                convert_angle_to_rad(np.array([50])),
                convert_angle_to_rad(np.array([100])),
                np.array([800]),
                np.array([0]),
                np.array([50]),
                np.array([2500.1]),
            ),
            (
                convert_angle_to_rad(np.array([91.631])),
                convert_angle_to_rad(np.array([91.562])),
                convert_angle_to_rad(np.array([91.562])),
                np.array([300]),
                np.array([0]),
                np.array([150]),
                np.array([1000.4]),
            ),
            (
                convert_angle_to_rad(np.array([101.607, 91.631])),
                convert_angle_to_rad(np.array([50, 91.562])),
                convert_angle_to_rad(np.array([100, 91.562])),
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
        self,
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

    def test_non_convergent_case_raises(self) -> None:
        """Test case adapted from tests cases provided by H. Ducloux"""
        angle_to_cable_tangent = convert_angle_to_rad(np.array([91.631]))
        angle_to_left_support = convert_angle_to_rad(np.array([124.224]))
        angle_to_right_support = convert_angle_to_rad(np.array([106.345]))

        with pytest.raises(ConvergenceError):
            compute_parameter__array(
                angle_to_cable_tangent,
                angle_to_left_support,
                angle_to_right_support,
                np.array([250]),
                np.array([0]),
                np.array([-50]),
            )

    def test_compute_parameter__scalar_ok(self) -> None:
        result = compute_parameter__scalar(
            convert_angle_to_rad(95.622),
            convert_angle_to_rad(75.776),
            convert_angle_to_rad(94.228),
            500,
            0,
            -50,
        )
        np.testing.assert_allclose(result, 2199.3, atol=1e-1)

    def test_compute_parameter__scalar_non_convergent_case_raises(
        self,
    ) -> None:
        angle_to_cable_tangent = convert_angle_to_rad(91.631)
        angle_to_left_support = convert_angle_to_rad(124.224)
        angle_to_right_support = convert_angle_to_rad(106.345)

        with pytest.raises(ConvergenceError):
            compute_parameter__scalar(
                angle_to_cable_tangent,
                angle_to_left_support,
                angle_to_right_support,
                250,
                0,
                -50,
            )


class TestComputeTangentAngle:
    # Test input validation

    def test_array__negative_parameter_raises(self) -> None:
        with pytest.raises(MechaphlowersValueError, match="Parameter"):
            compute_tangent_angle__array(
                # First value is wrong: should be positive
                np.array([-0.00001, 1200, 2500]),
                convert_angle_to_rad(np.array([75.776, 0, 50])),
                convert_angle_to_rad(np.array([94.228, 96.82, 100])),
                np.array([500, 400, 800]),
                np.array([0, 20, 0]),
                np.array([-50, 0, 50]),
            )

    def test_scalar__negative_left_angle_raises(self) -> None:
        with pytest.raises(
            MechaphlowersValueError, match="angle_to_left_support"
        ):
            compute_tangent_angle__scalar(
                1000.0,
                convert_angle_to_rad(-0.000001),
                convert_angle_to_rad(91.562),
                300.0,
                0.0,
                150.0,
            )

    def test_scalar__left_angle_greater_than_pi_raises(self) -> None:
        with pytest.raises(
            MechaphlowersValueError, match="angle_to_left_support"
        ):
            compute_tangent_angle__scalar(
                1000.0,
                np.pi + 0.00001,
                convert_angle_to_rad(91.562),
                300.0,
                0.0,
                150.0,
            )

    def test_scalar__negative_right_angle_raises(self) -> None:
        with pytest.raises(
            MechaphlowersValueError, match="angle_to_right_support"
        ):
            compute_tangent_angle__scalar(
                1000.0,
                convert_angle_to_rad(91.562),
                convert_angle_to_rad(-0.000001),
                300.0,
                0.0,
                150.0,
            )

    def test_scalar__right_angle_greater_than_pi_raises(self) -> None:
        with pytest.raises(
            MechaphlowersValueError, match="angle_to_right_support"
        ):
            compute_tangent_angle__scalar(
                1000.0,
                convert_angle_to_rad(91.562),
                np.pi + 0.00001,
                300.0,
                0.0,
                150.0,
            )

    def test_scalar__span_length_zero_raises(self) -> None:
        with pytest.raises(MechaphlowersValueError, match="span_length"):
            compute_tangent_angle__scalar(
                1000.0,
                convert_angle_to_rad(91.562),
                convert_angle_to_rad(91.562),
                0.0,
                0.0,
                150.0,
            )

    def test_scalar__negative_span_length_raises(self) -> None:
        with pytest.raises(MechaphlowersValueError, match="span_length"):
            compute_tangent_angle__scalar(
                1000.0,
                convert_angle_to_rad(91.562),
                convert_angle_to_rad(91.562),
                -0.00001,
                0.0,
                150.0,
            )

    def test_scalar__negative_input_height_raises(self) -> None:
        with pytest.raises(MechaphlowersValueError, match="input_height"):
            compute_tangent_angle__scalar(
                1200,
                convert_angle_to_rad(0),
                convert_angle_to_rad(96.82),
                400,
                -0.000001,
                0,
            )

    def test_scalar__input_height_and_distance_both_zero_raises(self) -> None:
        """
        input_height or distance must be provided (not zero, not nan).

        Indeed, distance = 0 means the sighting device is directly below
        the left hanging point, in which case input_height is required.
        """
        with pytest.raises(
            MechaphlowersValueError, match="input_height .* distance"
        ):
            compute_tangent_angle__scalar(
                1200,
                convert_angle_to_rad(0),
                convert_angle_to_rad(96.82),
                400,
                0,
                0,
            )

    def test_scalar__zero_angle_to_left_support_and_positive_distance_raises(
        self,
    ) -> None:
        with pytest.raises(
            MechaphlowersValueError,
            match="angle to the left support .* distance",
        ):
            compute_tangent_angle__scalar(
                1000.0,
                0,
                convert_angle_to_rad(91.562),
                300.0,
                0.0,
                150.0,
            )

    def test_scalar__zero_distance_and_positive_left_angle_raises(
        self,
    ) -> None:
        with pytest.raises(
            MechaphlowersValueError,
            match="distance .* angle to the left support",
        ):
            compute_tangent_angle__scalar(
                1200,
                convert_angle_to_rad(91.562),
                convert_angle_to_rad(96.82),
                400,
                20,
                0,
            )

    def test_scalar__nan_distance_and_positive_left_angle_raises(self) -> None:
        with pytest.raises(
            MechaphlowersValueError,
            match="distance .* angle to the left support",
        ):
            compute_tangent_angle__scalar(
                1200,
                convert_angle_to_rad(91.562),
                convert_angle_to_rad(96.82),
                400,
                20,
                np.nan,
            )

    def test_scalar__input_height_nan_ok_if_distance_provided(self) -> None:
        result = compute_tangent_angle__scalar(
            1000,
            convert_angle_to_rad(91.562),
            convert_angle_to_rad(91.562),
            300,
            np.nan,
            150,
        )
        np.testing.assert_allclose(
            result, convert_angle_to_rad(91.631), atol=1e-1
        )

    def test_scalar__distance_nan_ok_if_input_height_provided(self) -> None:
        result = compute_tangent_angle__scalar(
            1200,
            0,
            convert_angle_to_rad(96.82),
            400,
            20,
            np.nan,
        )
        np.testing.assert_allclose(
            result, convert_angle_to_rad(98.999), atol=1e-1
        )

    # Test results

    def test_compute_tangent_angle__array_ok(self) -> None:
        # Test cases derived from H. Ducloux prototype:
        # - left of left support - 1st test case of prototype doc
        # - at support - 2nd test case of prototype doc
        # - between supports - 3rd test case of prototype doc
        result = compute_tangent_angle__array(
            np.array([2200, 1200, 2500]),
            convert_angle_to_rad(np.array([75.776, 0, 50])),
            convert_angle_to_rad(np.array([94.228, 96.82, 100])),
            np.array([500, 400, 800]),
            np.array([0, 20, 0]),
            np.array([-50, 0, 50]),
        )

        np.testing.assert_allclose(
            result,
            convert_angle_to_rad(np.array([95.621, 98.999, 101.607])),
            atol=1e-1,
        )

    def test_compute_tangent_angle__scalar_ok(self) -> None:
        # Test case derived from H. Ducloux prototype:
        # between supports - 4th test case of prototype doc
        result = compute_tangent_angle__scalar(
            1000.0,
            convert_angle_to_rad(91.562),
            convert_angle_to_rad(91.562),
            300.0,
            0.0,
            150.0,
        )

        np.testing.assert_allclose(
            result, convert_angle_to_rad(91.631), atol=1e-1
        )

    # Test non-convergent cases

    def test_compute_tangent_angle__array_case_raises(self) -> None:
        # First columns of data are OK, second yield a ConvergenceError
        with pytest.raises(ConvergenceError, match="Solver did not converge"):
            compute_tangent_angle__array(
                np.array([2200, 800]),
                convert_angle_to_rad(np.array([75.776, 124.224])),
                convert_angle_to_rad(np.array([94.228, 106.345])),
                np.array([500, 250]),
                np.array([0, 0]),
                np.array([-50, -50]),
            )

    def test_compute_tangent_angle__scalar_case_raises(self) -> None:
        with pytest.raises(
            ConvergenceError,
            match="Found aberrant x - no solution",
        ):
            compute_tangent_angle__scalar(
                2400,
                convert_angle_to_rad(81.445),
                convert_angle_to_rad(90.521),
                600,
                0,
                200,
            )

    def test_compute_tangent_angle__scalar_case_raises_2(self) -> None:
        with pytest.raises(
            ConvergenceError,
            match="Found aberrant x - no solution",
        ):
            compute_tangent_angle__scalar(
                2400,
                convert_angle_to_rad(81.445),
                convert_angle_to_rad(90.521),
                600,
                0,
                200,
            )
