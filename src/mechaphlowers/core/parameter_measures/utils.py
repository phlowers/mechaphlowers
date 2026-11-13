import numpy as np
from numpy import ndarray

# TODO: mutualize with CableSpan.compute_x__m and .z_one_point?


def _lowest_point_coordinates(
    parameter: np.ndarray,
    span_length: np.ndarray,
    elevation_difference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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


def _catenary_equation_z(
    x: ndarray,
    parameter: ndarray,
) -> ndarray:
    # In cable frame
    return parameter * (np.cosh(x / parameter) - 1)
