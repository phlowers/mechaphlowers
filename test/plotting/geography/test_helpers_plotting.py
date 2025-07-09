# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import patch

import numpy as np


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

    @patch('mechaphlowers.data.geography.elevation.gps_to_elevation')
    def test_geo_info_from_gps(
        self,
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
        from mechaphlowers.entities.geography import geo_info_from_gps

        result = geo_info_from_gps(large_lats, large_lons)

        assert isinstance(result, dict)
        assert "latitude" in result
        assert "longitude" in result
        assert "elevation" in result
        assert "distance_to_next" in result
        assert "bearing_to_next" in result
        assert "direction_to_next" in result
        assert "lambert_93" in result

        # Verify GPS coordinates
        gps_lats = result["latitude"]
        gps_lons = result["longitude"]
        np.testing.assert_array_equal(gps_lats, large_lats)
        np.testing.assert_array_equal(gps_lons, large_lons)

        # Verify elevation
        np.testing.assert_array_equal(
            result["elevation"],
            np.array([35.0, 12.0, 173.0, 146.0, 143.0, 20.0]),
        )
        # Verify distance_to_next (should have n-1 elements for n GPS points)
        np.testing.assert_allclose(
            result["distance_to_next"],
            np.array(
                [
                    660478.379674,
                    277618.75794,
                    359858.313144,
                    735617.986473,
                    709464.274707,
                ]
            ),
            atol=1e-6,
        )

        # Verify bearing_to_next
        np.testing.assert_allclose(
            result["bearing_to_next"],
            np.array(
                [
                    158.26942811,
                    351.41470965,
                    229.35897765,
                    39.12509019,
                    261.22934784,
                ]
            ),
            atol=1e-6,
        )

        # Verify direction_to_next
        np.testing.assert_array_equal(
            result["direction_to_next"], np.array(['S', 'N', 'SW', 'NE', 'W'])
        )

        # Verify lambert_93 coordinates
        lambert_x, lambert_y = result["lambert_93"]
        np.testing.assert_allclose(
            lambert_x,
            np.array(
                [
                    652469.022709,
                    892390.221566,
                    842666.659187,
                    574357.820138,
                    1050362.695358,
                    355577.801578,
                ]
            ),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            lambert_y,
            np.array(
                [
                    6862035.25942,
                    6247035.256802,
                    6519924.366806,
                    6279642.972343,
                    6840899.647188,
                    6689723.102902,
                ]
            ),
            atol=1e-6,
        )
