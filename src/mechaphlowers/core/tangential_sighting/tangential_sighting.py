from typing import Tuple

import numpy as np

from mechaphlowers.entities.errors import ConvergenceError
from mechaphlowers.numeric.newton import newton_solver_wrapper
from mechaphlowers.utils import cotan


def _validate_inputs(
    angle_to_cable_tangent: np.ndarray,
    angle_to_left_support: np.ndarray,
    angle_to_right_support: np.ndarray,
    span_length: np.ndarray,
    input_height: np.ndarray,
    distance: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """NB: input angles are assumed to be in radians."""
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
    if (
        ((input_height == 0) | np.isnan(input_height))
        & ((distance == 0) | np.isnan(distance))
    ).any():
        raise ValueError(
            "input_height or distance must be provided (not zero, not nan)"
        )
    if (
        (input_height != 0)
        & ~np.isnan(input_height)
        & (distance != 0)
        & ~np.isnan(distance)
    ).any():
        raise ValueError(
            "input_height and distance can't be both provided (not zero, not nan)"
        )
    if (
        (angle_to_left_support == 0) & (distance != 0) & ~np.isnan(distance)
    ).any():
        raise ValueError(
            "If the angle to the left support is zero, the distance must be zero or nan."
        )
    if (
        ((distance == 0) | np.isnan(distance)) & (angle_to_left_support != 0)
    ).any():
        raise ValueError(
            "If the distance is zero or nan, the angle to the left support must be zero."
        )

    # Ensure all inputs are float arrays, because if they are integer arrays,
    # computation results may be truncated to integers.
    return (
        _convert_to_float_array(angle_to_cable_tangent),
        _convert_to_float_array(angle_to_left_support),
        _convert_to_float_array(angle_to_right_support),
        _convert_to_float_array(span_length),
        _convert_to_float_array(input_height),
        _convert_to_float_array(distance),
    )


def _convert_to_float_array(
    array: np.ndarray,
) -> np.ndarray:
    return np.array(array, dtype=np.float64)


def _prepare_angle_to_left_support_input(
    angle_to_left_support: np.ndarray, distance: np.ndarray
) -> np.ndarray:
    return np.where(
        distance > 0,
        2 * np.pi - angle_to_left_support,
        angle_to_left_support,
    )


# TODO: return partial results if only some inputs are invalid?
def compute_parameter__array(
    angle_to_cable_tangent: np.ndarray,
    angle_to_left_support: np.ndarray,
    angle_to_right_support: np.ndarray,
    span_length: np.ndarray,
    input_height: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    """Compute parameter using the tangential sighting method.

    Args:
        angle_to_cable_tangent: angle between the vertical and the tangent to the cable.
        angle_to_left_support: angle between the vertical and the left hanging point (very roughly the top of the left
            support).
        angle_to_right_support: angle between the vertical and the right hanging point (very roughly the top of the
            right support).
        span_length: length of the span.
        input_height: should only be provided if the sighting device is right below (or in rare cases above) the left
            support. This is the distance between the sighting device and the left hanging point. Else it should be
            zero.
        distance: distance between the left support and the sighting device (horizontal projection). Positive if the
            sighting device is between the supports, negative if it is left of the left support.

    All angles are in radians and are not oriented. They must be comprised between 0 and Pi.

    Returns:
        Computed parameters (as an array).

    Raises:
        ValueError: if any of the angles is not comprised between 0 and Pi;
            if angle_to_cable_tangent is zero,
            if span_length is negative or zero,
            if input_height is strictly negative,
            if both distance and input_height are zero (in which case we don't have enough information to compute
            the parameter),
            if both distance and input_height are provided, non-zero and not nan,
            if distance isn't zero, and angle_to_left_support is zero (geometrically impossible).

    """
    (
        angle_to_cable_tangent,
        angle_to_left_support,
        angle_to_right_support,
        span_length,
        input_height,
        distance,
    ) = _validate_inputs(
        angle_to_cable_tangent,
        angle_to_left_support,
        angle_to_right_support,
        span_length,
        input_height,
        distance,
    )

    angle_to_left_support = _prepare_angle_to_left_support_input(
        angle_to_left_support,
        distance,
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

    _check_result(result, span_length, elevation_difference, slope)

    return result


def _sighted_slope(
    x: np.ndarray,
    y: np.ndarray,
    corrected_height: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    return (y + corrected_height) / (x - distance)


def _compute_corrected_height(
    angle_to_left_support: np.ndarray,
    input_height: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    corrected_height = input_height.copy()
    mask = (input_height == 0) | np.isnan(input_height)
    if np.any(mask):
        corrected_height[mask] = -distance[mask] * cotan(
            angle_to_left_support[mask]
        )
    return corrected_height


def _compute_elevation_difference(
    angle_to_right_support: np.ndarray,
    span_length: np.ndarray,
    corrected_height: np.ndarray,
    distance: np.ndarray,
) -> np.ndarray:
    """Elevation difference between left and right hanging points.
    Positive if the right hanging point is higher than the left hanging point, negative otherwise."""
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
    parameter: np.ndarray,
    span_length: np.ndarray,
    elevation_difference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    parameter: np.ndarray,
    span_length: np.ndarray,
    elevation_difference: np.ndarray,
    slope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative to the left hanging point"""
    x_lowest, y_lowest = _lowest_point_coordinates(
        parameter, span_length, elevation_difference
    )
    x = x_lowest + parameter * np.asinh(slope)
    y = y_lowest + parameter * (np.cosh((x - x_lowest) / parameter) - 1)
    return x, y


def _check_result(
    computed_parameter: np.ndarray,
    span_length: np.ndarray,
    elevation_difference: np.ndarray,
    slope: np.ndarray,
) -> None:
    x, y = _tangent_point_coordinates(
        computed_parameter, span_length, elevation_difference, slope
    )
    if np.logical_or(x <= 0, x >= span_length).any():
        raise ConvergenceError(
            "Found aberrant x - no solution",
            origin="tangential_sighting",
        )
