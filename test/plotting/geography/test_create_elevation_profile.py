# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import plotly.graph_objects as go

from mechaphlowers.plotting.geography.create_elevation_profile import (
    create_elevation_profile,
)
from mechaphlowers.plotting.geography.type_hints import (
    SupportGeoInfo,
)


class TestCreateElevationProfile:
    """Test suite for create_elevation_profile function."""

    def setup_method(self):
        """Set up test data."""
        # Sample support geo info for testing
        self.sample_supports: list[SupportGeoInfo] = [
            {
                "gps": {"lat": 48.8566, "lon": 2.3522},  # Paris
                "elevation": 35.0,
                "distance_to_next": 100.0,
                "bearing_to_next": 180.5,
                "direction_to_next": "S",
                "lambert_93": {"x": 652469.02, "y": 6862035.26},
            },
            {
                "gps": {"lat": 43.2965, "lon": 5.3698},  # Marseille
                "elevation": 12.0,
                "distance_to_next": 200.0,
                "bearing_to_next": 315.2,
                "direction_to_next": "NW",
                "lambert_93": {"x": 892390.22, "y": 6247035.26},
            },
            {
                "gps": {"lat": 45.7640, "lon": 4.8357},  # Lyon
                "elevation": 173.0,
                "distance_to_next": 150.0,
                "bearing_to_next": 270.8,
                "direction_to_next": "W",
                "lambert_93": {"x": 842666.66, "y": 6519924.37},
            },
        ]

    def test_create_elevation_profile_basic_functionality(self):
        """Test basic functionality of create_elevation_profile."""
        fig = go.Figure()

        # Call the function
        create_elevation_profile(fig, self.sample_supports)

        # Verify that a trace was added
        assert len(fig.data) == 1

        # Get the added trace
        trace = fig.data[0]

        # Verify trace properties
        assert trace.type == "scatter"
        assert trace.mode == "lines+markers"
        assert trace.name == "Elevation Profile"
        assert trace.line.color == "#1f77b4"
        assert trace.line.width == 2
        assert trace.marker.size == 8
        assert trace.marker.color == "#1f77b4"

        # Verify data
        expected_cumulative_distance = np.array([0.0, 100.0, 300.0, 450.0])
        expected_elevations = np.array([35.0, 12.0, 173.0])
        expected_index_list = np.array([0, 1, 2])

        assert np.allclose(trace.x, expected_cumulative_distance)
        assert np.allclose(trace.y, expected_elevations)
        assert np.allclose(trace.customdata, expected_index_list)

        # Verify hover template
        expected_hover_template = "Marker: %{customdata}<br>Distance: %{x:.2f} km<br>Elevation: %{y:.2f} m<extra></extra>"
        assert trace.hovertemplate == expected_hover_template
