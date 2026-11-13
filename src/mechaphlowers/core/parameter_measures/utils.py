from typing import overload

import numpy as np

# TODO: mutualize with CableSpan.compute_x__m and .z_one_point?


@overload
def _lowest_point_coordinates(
    parameter: float,
    span_length: float,
    elevation_difference: float,
) -> tuple[float, float]: ...


@overload
def _lowest_point_coordinates(
    parameter: np.ndarray,
    span_length: np.ndarray,
    elevation_difference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]: ...


def _lowest_point_coordinates(
    parameter: np.ndarray | float,
    span_length: np.ndarray | float,
    elevation_difference: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Coordinates of the lowest point of the cable in the span frame.

    (Origin is the left hanging point.)"""
    x = (
        span_length / 2
        - np.asinh(
            elevation_difference
            / (2 * parameter * np.sinh(span_length / (2 * parameter)))
        )
        * parameter
    )
    z = -_catenary_equation_z(x, parameter)
    return x, z


@overload
def _catenary_equation_z(
    x: float,
    parameter: float,
) -> float: ...


@overload
def _catenary_equation_z(
    x: np.ndarray,
    parameter: np.ndarray,
) -> np.ndarray: ...


def _catenary_equation_z(
    x: np.ndarray | float,
    parameter: np.ndarray | float,
) -> np.ndarray | float:
    # In cable frame
    return parameter * (np.cosh(x / parameter) - 1)
