# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np

from mechaphlowers.data.geography.helpers import (
    gps_to_lambert93,
    lambert93_to_gps,
)

test_gps_coors = np.array(
    [
        [48.8566, 2.3522],  # Paris
        [43.2965, 5.3698],  # Marseille
        [45.7640, 4.8357],  # Lyon
        [43.6047, 1.4442],  # Toulouse
        [48.5734, 7.7521],  # Strasbourg
        [47.2184, -1.5536],  # Nantes
        [50.6292, 3.0573],  # Lille
        [44.8378, -0.5792],  # Bordeaux
    ],
    dtype=np.float64,
)

test_lambert_coords = np.array(
    [
        [652469.02, 6862035.26],  # Paris
        [892390.22, 6247035.26],  # Marseille
        [842666.66, 6519924.37],  # Lyon
        [574357.82, 6279642.97],  # Toulouse
        [1050362.70, 6840899.65],  # Strasbourg
        [355577.80, 6689723.10],  # Nantes
        [704061.15, 7059136.59],  # Lille
        [417241.54, 6421813.35],  # Bordeaux
    ],
    dtype=np.float64,
)


def test_gps_to_lambert93():
    """Test GPS to Lambert 93 conversion with multiple coordinates using matrices."""

    latitudes = test_gps_coors[:, 0]
    longitudes = test_gps_coors[:, 1]

    x_coords, y_coords = gps_to_lambert93(latitudes, longitudes)

    assert isinstance(x_coords, np.ndarray)
    assert isinstance(y_coords, np.ndarray)
    assert x_coords.shape == (8,)
    assert y_coords.shape == (8,)
    assert x_coords.dtype == np.float64
    assert y_coords.dtype == np.float64

    assert np.allclose(x_coords, test_lambert_coords[:, 0], atol=1e-08)
    assert np.allclose(y_coords, test_lambert_coords[:, 1], atol=1e-08)


def test_lambert93_to_gps():
    """Test Lambert 93 to GPS conversion with multiple coordinates using matrices."""

    x_coords = test_lambert_coords[:, 0]
    y_coords = test_lambert_coords[:, 1]

    latitudes, longitudes = lambert93_to_gps(x_coords, y_coords)

    assert isinstance(latitudes, np.ndarray)
    assert isinstance(longitudes, np.ndarray)
    assert latitudes.shape == (8,)
    assert longitudes.shape == (8,)
    assert latitudes.dtype == np.float64
    assert longitudes.dtype == np.float64

    assert np.allclose(latitudes, test_gps_coors[:, 0], atol=1e-6)
    assert np.allclose(longitudes, test_gps_coors[:, 1], atol=1e-6)
