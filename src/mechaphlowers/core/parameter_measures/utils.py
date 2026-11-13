import numpy as np


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
    z = -parameter * (np.cosh(x / parameter) - 1)
    return x, z
