# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Plotting utility functions for data visualization and layout configuration."""

from __future__ import annotations

import numpy as np

from mechaphlowers.core.geometry.points import Points

# Minimum normalized aspect ratio value to avoid zero-sized axes in Plotly
_ASPECT_EPSILON: float = 1e-4


def compute_aspect_ratio(
    *points_objects: Points,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    z_scale: float = 1.0,
) -> dict[str, float]:
    """Compute an aspect ratio dictionary for Plotly 3D plots based on data coordinates.

    This function analyzes the spatial extent of multiple Points objects (typically spans,
    supports, and insulators from a power line section) and computes normalized aspect ratios
    that respect the actual data ranges while allowing custom scaling per axis.

    The algorithm:

    1. Concatenates all points from all Points objects into a single array
    2. Computes min/max for each axis (x, y, z)
    3. Calculates the range (max - min) for each axis
    4. Normalizes each range by dividing by the maximum of all three ranges
    5. Applies custom scaling factors to each normalized ratio
    6. Returns a dict suitable for Plotly's aspectratio parameter

    Args:
        *points_objects: Variable number of Points objects to analyze. Typically these
            come from PlotEngine.get_points_for_plot() which returns (spans, supports, insulators).
        x_scale (float, optional): Scaling factor for the x-axis ratio. Defaults to 1.0.
        y_scale (float, optional): Scaling factor for the y-axis ratio. Defaults to 1.0.
        z_scale (float, optional): Scaling factor for the z-axis ratio. Defaults to 1.0.
            Common use case: z_scale=10 to exaggerate altitude dimension for better visibility.

    Returns:
        dict[str, float]: A dictionary with keys 'x', 'y', 'z' containing the normalized
            aspect ratios scaled by the provided factors. Each value is a float in the range
            [0.0001, scale_factor] approximately (exact range depends on data).

    Raises:
        ValueError: If no Points objects are provided, or if all points are NaN.
        ValueError: If scale factors are not positive.

    Examples:
        >>> from mechaphlowers.plotting.plot import PlotEngine
        >>> plot_engine = PlotEngine.builder_from_balance_engine(balance_engine)
        >>> spans, supports, insulators = plot_engine.get_points_for_plot()
        >>>
        >>> # Compute aspect ratio with default scaling (equal for all axes)
        >>> aspect = compute_aspect_ratio(spans, supports, insulators)
        >>> print(aspect)  # {'x': 0.45, 'y': 0.30, 'z': 1.0}
        >>>
        >>> # Compute aspect ratio with z-axis exaggeration (common for altitude visualization)
        >>> aspect = compute_aspect_ratio(spans, supports, insulators, z_scale=10)
        >>> print(aspect)  # {'x': 0.45, 'y': 0.30, 'z': 10.0}

    See Also:
        PlotEngine.preview_line3d: Plotting method that can use this function.
        Points: The coordinate class used to represent geometric data.
    """
    # Input validation
    if not points_objects:
        raise ValueError("At least one Points object must be provided")

    if x_scale <= 0 or y_scale <= 0 or z_scale <= 0:
        raise ValueError(
            f"Scale factors must be positive; got x_scale={x_scale}, y_scale={y_scale}, z_scale={z_scale}"
        )

    # Concatenate all points from all Points objects
    all_points_list = []
    for points_obj in points_objects:
        if not isinstance(points_obj, Points):
            raise TypeError(
                f"Expected Points object, got {type(points_obj).__name__}"
            )
        # Get flat array of shape (N, 3) where each row is [x, y, z]
        points_array = points_obj.points(stack=False)
        all_points_list.append(points_array)

    all_points = np.vstack(all_points_list)

    if all_points.size == 0:
        raise ValueError(
            "At least one Points object must contain at least one point to compute aspect ratio"
        )

    # Extract x, y, z coordinates
    xs = all_points[:, 0]
    ys = all_points[:, 1]
    zs = all_points[:, 2]

    # Compute ranges using nanmin/nanmax to handle NaN values
    x_range = np.nanmax(xs) - np.nanmin(xs)
    y_range = np.nanmax(ys) - np.nanmin(ys)
    z_range = np.nanmax(zs) - np.nanmin(zs)

    # Handle edge case where all values in an axis are NaN
    if np.isnan(x_range) or np.isnan(y_range) or np.isnan(z_range):
        raise ValueError(
            "Cannot compute aspect ratio because at least one axis has only NaN values"
        )

    # Normalize by the maximum range
    max_range = max(x_range, y_range, z_range)
    if max_range == 0:
        raise ValueError(
            "Data has zero spatial extent; cannot compute aspect ratio"
        )

    # Compute normalized ranges and clamp zero-extent axes to a small epsilon
    norm_x = x_range / max_range if x_range > 0 else _ASPECT_EPSILON
    norm_y = y_range / max_range if y_range > 0 else _ASPECT_EPSILON
    norm_z = z_range / max_range if z_range > 0 else _ASPECT_EPSILON

    aspect_x = norm_x * x_scale
    aspect_y = norm_y * y_scale
    aspect_z = norm_z * z_scale

    return {"x": float(aspect_x), "y": float(aspect_y), "z": float(aspect_z)}
