import numpy as np

from mechaphlowers.entities.errors import ConvergenceError
from mechaphlowers.numeric.newton import newton_solver_wrapper
from mechaphlowers.utils import acotan, cotan

# TODO: complete typing and docs


def _check_inputs(
    angle_to_cable_tangent: np.ndarray,
    angle_to_left_support: np.ndarray,
    angle_to_right_support: np.ndarray,
    span_length: np.ndarray,
    input_height: np.ndarray,
    distance: np.ndarray,
) -> None:
    """Radians"""
    for angle, variable_name in zip(
        [
            angle_to_cable_tangent,
            angle_to_left_support,
            angle_to_right_support,
        ],
        [
            "angle_to_cable_tangent",
            "angle_to_left_support",
            "angle_to_right_support",
        ],
    ):
        if np.logical_or(angle < 0, angle > np.pi).any():
            raise ValueError(
                f"All input angles must be between 0 and pi radians, got {variable_name}={angle} rad."
            )
    if (angle_to_cable_tangent == 0).any():
        raise ValueError(
            f"Can't compute a parameter if angle_to_cable_tangent == 0. Got {angle_to_cable_tangent=}"
        )
    if (span_length <= 0).any():
        raise ValueError(f"span_length must be positive, got {span_length}.")
    if (input_height < 0).any():
        raise ValueError(
            f"input_height must be strictly positive, got {input_height}."
        )
    if np.logical_and(input_height == 0, distance == 0).any():
        raise ValueError("input_height and distance can't both be zero")
    if np.logical_and(input_height != 0, distance != 0).any():
        raise ValueError("input_height and distance can't be both provided")
    if np.logical_and(distance != 0, angle_to_left_support == 0).any():
        raise ValueError(
            "angle to left support can't be zero if distance isn't zero"
        )


def _prepare_angle_to_left_support_input(
    angle_to_left_support: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    return np.where(
        distance > 0, 2 * np.pi - angle_to_left_support, angle_to_left_support
    )


def compute_parameter__array(
    angle_to_cable_tangent: np.ndarray,
    angle_to_left_support: np.ndarray,
    angle_to_right_support: np.ndarray,
    span_length: np.ndarray,
    input_height: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    """Radians"""
    # TODO: accept nans ? None ? for distance or input_height
    _check_inputs(
        angle_to_cable_tangent,
        angle_to_left_support,
        angle_to_right_support,
        span_length,
        input_height,
        distance,
    )

    angle_to_left_support = _prepare_angle_to_left_support_input(
        angle_to_left_support, distance
    )

    corrected_height = _compute_corrected_height(
        angle_to_left_support,
        input_height,
        distance,
    )

    elevation_difference = _compute_elevation_difference(
        angle_to_right_support,
        span_length,
        corrected_height,
        distance,
    )

    slope = cotan(angle_to_cable_tangent)

    approx_parameter = approx_parameter_using_parabola(
        span_length,
        corrected_height,
        distance,
        slope,
        elevation_difference,
    )

    def f(parameter):
        x_tangent, y_tangent = _tangent_point_coordinates(
            parameter,
            span_length,
            elevation_difference,
            slope,
        )
        sighted_slope = _sighted_slope(
            x_tangent, y_tangent, corrected_height, distance
        )
        return sighted_slope - slope

    def f_prime_approx(parameter):
        return f(parameter + 1) - f(parameter)

    result = newton_solver_wrapper(
        f,
        x0=approx_parameter,
        fprime=f_prime_approx,
        args=(),
        caller_name="tangential_sighting",
    )

    _check_result(
        result,
        span_length,
        distance,
        corrected_height,
        elevation_difference,
        slope,
    )

    return result


def _sighted_slope(x, y, corrected_height, distance):
    return (y + corrected_height) / (x - distance)


def _compute_corrected_height(angle_to_left_support, input_height, distance):
    corrected_height = input_height.copy()
    corrected_height[input_height == 0] = -distance * cotan(
        angle_to_left_support
    )
    return corrected_height


def _compute_elevation_difference(
    angle_to_right_support, span_length, corrected_height, distance
):
    return (span_length - distance) * cotan(
        angle_to_right_support
    ) - corrected_height


def approx_parameter_using_parabola(
    span_length: np.ndarray,
    corrected_height: np.ndarray,
    distance: np.ndarray,
    slope: np.ndarray,
    elevation_difference: np.ndarray,
) -> np.ndarray:
    """Approx using parabola equation"""
    a = corrected_height + distance * slope
    b = (corrected_height + elevation_difference) - (
        span_length - distance
    ) * slope
    sag = ((a + b) / 2 + np.sqrt(a * b)) / 2
    return span_length * span_length / (8 * sag)


def _lowest_point_coordinates(
    parameter,
    span_length: np.ndarray,
    elevation_difference: np.ndarray,
):
    """Relative to the left hanging point"""
    x = (
        span_length / 2
        - np.asinh(
            elevation_difference
            / (2 * parameter * np.sinh(span_length / (2 * parameter)))
        )
        * parameter
    )
    y = -parameter * (np.cosh(x / parameter) - 1)
    return x, y


def _tangent_point_coordinates(
    parameter,
    span_length,
    elevation_difference,
    slope,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative to the left hanging point"""
    x_lowest, y_lowest = _lowest_point_coordinates(
        parameter, span_length, elevation_difference
    )
    x = x_lowest + parameter * np.asinh(slope)
    y = y_lowest + parameter * (np.cosh((x - x_lowest) / parameter) - 1)
    return x, y


def _check_result(
    computed_parameter,
    span_length,
    distance,
    corrected_height,
    elevation_difference,
    slope,
) -> None:
    x, y = _tangent_point_coordinates(
        computed_parameter, span_length, elevation_difference, slope
    )
    if x <= 0 or x >= span_length:
        raise ConvergenceError(
            "Found aberrant x - no solution",
            origin="tangential_sighting",
        )
    computed_slope = _sighted_slope(x, y, corrected_height, distance)
    computed_angle_tangent = acotan(computed_slope)
    if (computed_angle_tangent < 0 or computed_angle_tangent > np.pi).any():
        raise ConvergenceError(
            "Found aberrant angle tangent - no solution",
            origin="tangential_sighting",
        )
