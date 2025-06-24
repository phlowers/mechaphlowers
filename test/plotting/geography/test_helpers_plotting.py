# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import patch

import numpy as np

from mechaphlowers.plotting.geography.helpers import geo_info_from_gps


class TestGeoInfoFromGps:
    """Test suite for geo_info_from_gps function."""

    def setup_method(self):
        """Set up test data."""
        # Sample GPS coordinates for testing (French cities)
        self.lats = np.array(
            [48.8566, 43.2965, 45.7640, 43.6047]
        )  # Paris, Marseille, Lyon, Toulouse
        self.lons = np.array(
            [2.3522, 5.3698, 4.8357, 1.4442]
        )  # Paris, Marseille, Lyon, Toulouse

    @patch('mechaphlowers.plotting.geography.helpers.gps_to_elevation')
    @patch('mechaphlowers.plotting.geography.helpers.gps_to_lambert93')
    @patch('mechaphlowers.plotting.geography.helpers.haversine')
    @patch('mechaphlowers.plotting.geography.helpers.gps_to_bearing')
    @patch('mechaphlowers.plotting.geography.helpers.bearing_to_direction')
    def test_geo_info_from_gps(
        self,
        mock_bearing_to_direction,
        mock_gps_to_bearing,
        mock_haversine,
        mock_gps_to_lambert93,
        mock_gps_to_elevation,
    ):
        """Test geo_info_from_gps with a larger dataset."""
        large_lats = np.array(
            [48.8566, 43.2965, 45.7640, 43.6047, 48.5734, 47.2184]
        )  # Paris, Marseille, Lyon, Toulouse, Strasbourg, Nantes
        large_lons = np.array(
            [2.3522, 5.3698, 4.8357, 1.4442, 7.7521, -1.5536]
        )

        # Mock return values for 6 points
        mock_gps_to_elevation.return_value = np.array(
            [35.0, 12.0, 173.0, 146.0, 143.0, 20.0]
        )
        mock_gps_to_lambert93.return_value = (
            np.array(
                [
                    652469.02,
                    892390.22,
                    842666.66,
                    574357.82,
                    1050362.70,
                    355577.80,
                ]
            ),
            np.array(
                [
                    6862035.26,
                    6247035.26,
                    6519924.37,
                    6279642.97,
                    6840899.65,
                    6689723.10,
                ]
            ),
        )
        mock_haversine.return_value = np.array(
            [661.2, 302.8, 380.5, 450.3, 520.1]
        )
        mock_gps_to_bearing.return_value = np.array(
            [180.5, 315.2, 270.8, 45.6, 90.3]
        )
        mock_bearing_to_direction.return_value = np.array(
            ['S', 'NW', 'W', 'NE', 'E']
        )

        result = geo_info_from_gps(large_lats, large_lons)

        assert isinstance(result, dict)
        assert "gps" in result
        assert "elevation" in result
        assert "distance_to_next" in result
        assert "bearing_to_next" in result
        assert "direction_to_next" in result
        assert "lambert_93" in result

        # Verify GPS coordinates
        gps_lats, gps_lons = result["gps"]
        np.testing.assert_array_equal(gps_lats, large_lats)
        np.testing.assert_array_equal(gps_lons, large_lons)

        # Verify elevation
        np.testing.assert_array_equal(
            result["elevation"],
            np.array([35.0, 12.0, 173.0, 146.0, 143.0, 20.0]),
        )

        # Verify distance_to_next (should have n-1 elements for n GPS points)
        np.testing.assert_array_equal(
            result["distance_to_next"],
            np.array([661.2, 302.8, 380.5, 450.3, 520.1]),
        )

        # Verify bearing_to_next
        np.testing.assert_array_equal(
            result["bearing_to_next"],
            np.array([180.5, 315.2, 270.8, 45.6, 90.3]),
        )

        # Verify direction_to_next
        np.testing.assert_array_equal(
            result["direction_to_next"], np.array(['S', 'NW', 'W', 'NE', 'E'])
        )

        # Verify lambert_93 coordinates
        lambert_x, lambert_y = result["lambert_93"]
        np.testing.assert_array_equal(
            lambert_x,
            np.array(
                [
                    652469.02,
                    892390.22,
                    842666.66,
                    574357.82,
                    1050362.70,
                    355577.80,
                ]
            ),
        )
        np.testing.assert_array_equal(
            lambert_y,
            np.array(
                [
                    6862035.26,
                    6247035.26,
                    6519924.37,
                    6279642.97,
                    6840899.65,
                    6689723.10,
                ]
            ),
        )
