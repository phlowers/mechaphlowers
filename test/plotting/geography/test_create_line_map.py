# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest

try:
    import plotly.graph_objects as go  # type: ignore[import-untyped]
except ImportError:
    pytest.skip("plotly is not installed", allow_module_level=True)
from mechaphlowers.entities.geography import (
    SupportGeoInfo,
)
from mechaphlowers.plotting.maps import plot_line_map


class TestCreateLineMap:
    """Test suite for plot_line_map function."""

    def setup_method(self):
        """Set up test data."""
        # Sample support geo info for testing - using arrays as per the actual implementation
        self.sample_supports: SupportGeoInfo = {
            "latitude": np.array([48.8566, 43.2965, 45.7640]),
            "longitude": np.array([2.3522, 5.3698, 4.8357]),
            "elevation": np.array([35.0, 12.0, 173.0]),
            "distance_to_next": np.array([100.0, 200.0, 150.0]),
            "bearing_to_next": np.array([180.5, 315.2, 270.8]),
            "direction_to_next": np.array(["S", "NW", "W"]),
            "lambert_93": (
                np.array([652469.02, 892390.22, 842666.66]),
                np.array([6862035.26, 6247035.26, 6519924.37]),
            ),
        }

    def test_plot_line_map(self):
        """Test basic functionality of plot_line_map."""
        fig = go.Figure()

        # Call the function
        plot_line_map(fig, self.sample_supports)

        # Verify that traces were added to the figure
        assert len(fig.data) == 1  # Single combined trace with lines+markers

        # Verify the combined trace
        trace = fig.data[0]
        assert trace.type == 'scattermapbox'
        assert trace.mode == 'lines+markers'

        # Verify coordinates
        expected_lats = np.array([48.8566, 43.2965, 45.7640])
        expected_lons = np.array([2.3522, 5.3698, 4.8357])
        assert np.allclose(trace.lat, expected_lats)
        assert np.allclose(trace.lon, expected_lons)

        # Verify line properties
        assert trace.line.color == '#666666'
        assert trace.line.width == 2

        # Verify marker properties
        assert trace.marker.size == 10
        assert trace.marker.color == '#FF0000'
        assert trace.marker.opacity == 0.8

        # Verify hover properties
        assert trace.hoverinfo == "text"
        assert trace.hovertemplate == '%{text}<extra></extra>'
        assert trace.name == "Power Line Supports"
        assert trace.showlegend is True

        # Verify hover texts
        assert len(trace.text) == 3
        for i, hover_text in enumerate(trace.text):
            assert f"Support {i}" in hover_text
            assert f"Lat: {expected_lats[i]:.6f}" in hover_text
            assert f"Lon: {expected_lons[i]:.6f}" in hover_text
            assert (
                f"Distance to next: {self.sample_supports['distance_to_next'][i]:.2f} km"
                in hover_text
            )
            assert (
                f"Bearing to next: {self.sample_supports['bearing_to_next'][i]:.1f}°"
                in hover_text
            )
            assert (
                f"Elevation: {self.sample_supports['elevation'][i]:.2f} m"
                in hover_text
            )

        # Verify layout updates
        assert fig.layout.mapbox is not None
        assert fig.layout.mapbox.style == "carto-positron"
        assert fig.layout.mapbox.center.lat == expected_lats.mean()
        assert fig.layout.mapbox.center.lon == expected_lons.mean()
        assert fig.layout.mapbox.zoom == 11
        assert fig.layout.margin.r == 0
        assert fig.layout.margin.t == 0
        assert fig.layout.margin.l == 0
        assert fig.layout.margin.b == 0
        assert fig.layout.height == 700
        assert fig.layout.width == 1200
        assert fig.layout.showlegend is True
