# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import Mock, patch

import numpy as np

from mechaphlowers.data.geography.elevation import OpenElevationService


@patch('mechaphlowers.data.geography.elevation.requests.post')
def test_gps_to_elevation(mock_post):
    """Test OpenElevationService."""

    # Mock successful API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "results": [
            {"elevation": 35.0},  # Paris elevation
            {"elevation": 12.0},  # Marseille elevation
            {"elevation": 173.0},  # Lyon elevation
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    # Test coordinates for French cities
    lat = np.array(
        [48.8566, 43.2965, 45.7640], dtype=np.float64
    )  # Paris, Marseille, Lyon
    lon = np.array([2.3522, 5.3698, 4.8357], dtype=np.float64)

    service = OpenElevationService()
    elevations = service.get_elevation(lat, lon)

    # Check that the function was called with correct parameters
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert call_args[0][0] == "https://api.open-elevation.com/api/v1/lookup"

    # Check the payload format
    payload = call_args[1]['json']
    assert 'locations' in payload

    # Check return values
    assert isinstance(elevations, np.ndarray)
    assert elevations.shape == (3,)
    assert elevations.dtype == np.float64
    assert np.allclose(elevations, np.array([35.0, 12.0, 173.0]))
