# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import plotly.graph_objects as go

from mechaphlowers.entities.geography import (
    SupportGeoInfo,
)
from mechaphlowers.plotting.elevation import (
    plot_elevation_profile,
)


class TestCreateElevationProfile:
    """Test suite for plot_elevation_profile function."""

    def setup_method(self):
        """Set up test data."""
        # Sample support geo info for testing - single SupportGeoInfo object with arrays
        self.sample_supports: SupportGeoInfo = {
            "gps": (
                np.array(
                    [48.8566, 43.2965, 45.7640]
                ),  # lats: Paris, Marseille, Lyon
                np.array(
                    [2.3522, 5.3698, 4.8357]
                ),  # lons: Paris, Marseille, Lyon
            ),
            "elevation": np.array([35.0, 12.0, 173.0]),
            "distance_to_next": np.array([100.0, 200.0, 150.0]),
            "bearing_to_next": np.array([180.5, 315.2, 270.8]),
            "direction_to_next": np.array(["S", "NW", "W"]),
            "lambert_93": (
                np.array([652469.02, 892390.22, 842666.66]),  # x coordinates
                np.array(
                    [6862035.26, 6247035.26, 6519924.37]
                ),  # y coordinates
            ),
        }

    def test_plot_elevation_profile_basic_functionality(self):
        """Test basic functionality of plot_elevation_profile."""
        fig = go.Figure()

        # Call the function
        plot_elevation_profile(fig, self.sample_supports)

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
        expected_cumulative_distance = np.array([100.0, 300.0, 450.0])
        expected_elevations = np.array([35.0, 12.0, 173.0])
        expected_index_list = np.array([0, 1, 2])

        assert np.allclose(trace.x, expected_cumulative_distance)
        assert np.allclose(trace.y, expected_elevations)
        assert np.allclose(trace.customdata, expected_index_list)

        # Verify hover template
        expected_hover_template = "Marker: %{customdata}<br>Distance: %{x:.2f} km<br>Elevation: %{y:.2f} m<extra></extra>"
        assert trace.hovertemplate == expected_hover_template
