# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import patch

import numpy as np
import pytest

from mechaphlowers.plotting.geography.helpers import geo_info_from_gps
from mechaphlowers.plotting.geography.type_hints import (
    GpsPoint,
)


class TestGeoInfoFromGps:
    """Test suite for geo_info_from_gps function."""

    def setup_method(self):
        """Set up test data."""
        # Sample GPS points for testing (French cities)
        self.gps_points: list[GpsPoint] = [
            {"lat": 48.8566, "lon": 2.3522},  # Paris
            {"lat": 43.2965, "lon": 5.3698},  # Marseille
            {"lat": 45.7640, "lon": 4.8357},  # Lyon
            {"lat": 43.6047, "lon": 1.4442},  # Toulouse
        ]

    @patch('mechaphlowers.plotting.geography.helpers.gps_to_elevation')
    @patch('mechaphlowers.plotting.geography.helpers.gps_to_lambert93')
    @patch('mechaphlowers.plotting.geography.helpers.haversine')
    @patch('mechaphlowers.plotting.geography.helpers.gps_to_bearing')
    @patch('mechaphlowers.plotting.geography.helpers.bearing_to_direction')
    def test_geo_info_from_gps_basic_functionality(
        self,
        mock_bearing_to_direction,
        mock_gps_to_bearing,
        mock_haversine,
        mock_gps_to_lambert93,
        mock_gps_to_elevation,
    ):
        """Test basic functionality of geo_info_from_gps with mocked dependencies."""
        # Mock return values
        mock_gps_to_elevation.return_value = np.array(
            [35.0, 12.0, 173.0, 146.0]
        )
        mock_gps_to_lambert93.return_value = (
            np.array([652469.02, 892390.22, 842666.66, 574357.82]),
            np.array([6862035.26, 6247035.26, 6519924.37, 6279642.97]),
        )
        mock_haversine.return_value = np.array([661.2, 302.8, 380.5])
        mock_gps_to_bearing.return_value = np.array([180.5, 315.2, 270.8])
        mock_bearing_to_direction.return_value = np.array(['S', 'NW', 'W'])

        # Call the function
        result = geo_info_from_gps(self.gps_points)

        # Verify the result structure
        assert isinstance(result, list)
        assert len(result) == 3  # Should have n-1 elements for n GPS points

        # Verify the first result
        first_result = result[0]
        assert first_result["gps"] == self.gps_points[0]
        assert first_result["distance_to_next"] == 661.2
        assert first_result["bearing_to_next"] == 180.5
        assert first_result["direction_to_next"] == 'S'
        assert first_result["elevation"] == 35.0
        assert first_result["lambert_93"]["x"] == 652469.02
        assert first_result["lambert_93"]["y"] == 6862035.26

        # Verify the second result
        second_result = result[1]
        assert second_result["gps"] == self.gps_points[1]
        assert second_result["distance_to_next"] == 302.8
        assert second_result["bearing_to_next"] == 315.2
        assert second_result["direction_to_next"] == 'NW'
        assert second_result["elevation"] == 12.0
        assert second_result["lambert_93"]["x"] == 892390.22
        assert second_result["lambert_93"]["y"] == 6247035.26

        # Verify the third result
        third_result = result[2]
        assert third_result["gps"] == self.gps_points[2]
        assert third_result["distance_to_next"] == 380.5
        assert third_result["bearing_to_next"] == 270.8
        assert third_result["direction_to_next"] == 'W'
        assert third_result["elevation"] == 173.0
        assert third_result["lambert_93"]["x"] == 842666.66
        assert third_result["lambert_93"]["y"] == 6519924.37

        # Verify that the mocked functions were called with correct parameters
        mock_gps_to_elevation.assert_called_once()
        mock_gps_to_lambert93.assert_called_once()
        mock_haversine.assert_called_once()
        mock_gps_to_bearing.assert_called_once()
        mock_bearing_to_direction.assert_called_once()

        # Verify the parameters passed to mocked functions
        lats = np.array([point["lat"] for point in self.gps_points])
        lons = np.array([point["lon"] for point in self.gps_points])

        np.testing.assert_array_equal(
            mock_gps_to_elevation.call_args[0][0], lats
        )
        np.testing.assert_array_equal(
            mock_gps_to_elevation.call_args[0][1], lons
        )

        np.testing.assert_array_equal(
            mock_gps_to_lambert93.call_args[0][0], lats
        )
        np.testing.assert_array_equal(
            mock_gps_to_lambert93.call_args[0][1], lons
        )

        np.testing.assert_array_equal(
            mock_haversine.call_args[0][0], lats[:-1]
        )
        np.testing.assert_array_equal(
            mock_haversine.call_args[0][1], lons[:-1]
        )
        np.testing.assert_array_equal(mock_haversine.call_args[0][2], lats[1:])
        np.testing.assert_array_equal(mock_haversine.call_args[0][3], lons[1:])

        np.testing.assert_array_equal(
            mock_gps_to_bearing.call_args[0][0], lats[:-1]
        )
        np.testing.assert_array_equal(
            mock_gps_to_bearing.call_args[0][1], lons[:-1]
        )
        np.testing.assert_array_equal(
            mock_gps_to_bearing.call_args[0][2], lats[1:]
        )
        np.testing.assert_array_equal(
            mock_gps_to_bearing.call_args[0][3], lons[1:]
        )

    def test_geo_info_from_gps_single_point(self):
        """Test geo_info_from_gps with a single GPS point."""
        single_point = [{"lat": 48.8566, "lon": 2.3522}]  # Paris only

        result = geo_info_from_gps(single_point)

        # Should return empty list for single point
        assert isinstance(result, list)
        assert len(result) == 0

    def test_geo_info_from_gps_empty_list(self):
        """Test geo_info_from_gps with an empty list."""
        result = geo_info_from_gps([])

        assert isinstance(result, list)
        assert len(result) == 0

    @patch('mechaphlowers.plotting.geography.helpers.gps_to_elevation')
    @patch('mechaphlowers.plotting.geography.helpers.gps_to_lambert93')
    @patch('mechaphlowers.plotting.geography.helpers.haversine')
    @patch('mechaphlowers.plotting.geography.helpers.gps_to_bearing')
    @patch('mechaphlowers.plotting.geography.helpers.bearing_to_direction')
    def test_geo_info_from_gps_two_points(
        self,
        mock_bearing_to_direction,
        mock_gps_to_bearing,
        mock_haversine,
        mock_gps_to_lambert93,
        mock_gps_to_elevation,
    ):
        """Test geo_info_from_gps with exactly two GPS points."""
        two_points = [
            {"lat": 48.8566, "lon": 2.3522},  # Paris
            {"lat": 43.2965, "lon": 5.3698},  # Marseille
        ]

        # Mock return values
        mock_gps_to_elevation.return_value = np.array([35.0, 12.0])
        mock_gps_to_lambert93.return_value = (
            np.array([652469.02, 892390.22]),
            np.array([6862035.26, 6247035.26]),
        )
        mock_haversine.return_value = np.array([661.2])
        mock_gps_to_bearing.return_value = np.array([180.5])
        mock_bearing_to_direction.return_value = np.array(['S'])

        result = geo_info_from_gps(two_points)

        assert isinstance(result, list)
        assert len(result) == 1  # Should have 1 element for 2 GPS points

        # Verify the single result
        first_result = result[0]
        assert first_result["gps"] == two_points[0]
        assert first_result["distance_to_next"] == 661.2
        assert first_result["bearing_to_next"] == 180.5
        assert first_result["direction_to_next"] == 'S'
        assert first_result["elevation"] == 35.0
        assert first_result["lambert_93"]["x"] == 652469.02
        assert first_result["lambert_93"]["y"] == 6862035.26

    @patch('mechaphlowers.plotting.geography.helpers.gps_to_elevation')
    @patch('mechaphlowers.plotting.geography.helpers.gps_to_lambert93')
    @patch('mechaphlowers.plotting.geography.helpers.haversine')
    @patch('mechaphlowers.plotting.geography.helpers.gps_to_bearing')
    @patch('mechaphlowers.plotting.geography.helpers.bearing_to_direction')
    def test_geo_info_from_gps_large_dataset(
        self,
        mock_bearing_to_direction,
        mock_gps_to_bearing,
        mock_haversine,
        mock_gps_to_lambert93,
        mock_gps_to_elevation,
    ):
        """Test geo_info_from_gps with a larger dataset."""
        large_dataset = [
            {"lat": 48.8566, "lon": 2.3522},  # Paris
            {"lat": 43.2965, "lon": 5.3698},  # Marseille
            {"lat": 45.7640, "lon": 4.8357},  # Lyon
            {"lat": 43.6047, "lon": 1.4442},  # Toulouse
            {"lat": 48.5734, "lon": 7.7521},  # Strasbourg
            {"lat": 47.2184, "lon": -1.5536},  # Nantes
        ]

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

        result = geo_info_from_gps(large_dataset)

        assert isinstance(result, list)
        assert len(result) == 5  # Should have n-1 elements for n GPS points

        # Verify all results have the correct structure
        for i, item in enumerate(result):
            assert item["gps"] == large_dataset[i]
            assert "distance_to_next" in item
            assert "bearing_to_next" in item
            assert "direction_to_next" in item
            assert "elevation" in item
            assert "lambert_93" in item
            assert "x" in item["lambert_93"]
            assert "y" in item["lambert_93"]

    def test_geo_info_from_gps_invalid_input(self):
        """Test geo_info_from_gps with invalid input types."""
        # Test with None
        with pytest.raises(TypeError):
            geo_info_from_gps(None)  # type: ignore

        # Test with wrong data structure
        invalid_points = [
            {"latitude": 48.8566, "longitude": 2.3522}
        ]  # Wrong keys
        with pytest.raises(KeyError):
            geo_info_from_gps(invalid_points)  # type: ignore

        # Test with missing lat/lon
        invalid_points2 = [{"lat": 48.8566}]  # Missing lon
        with pytest.raises(KeyError):
            geo_info_from_gps(invalid_points2)  # type: ignore
