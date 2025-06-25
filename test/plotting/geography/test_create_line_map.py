# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import plotly.graph_objects as go

from mechaphlowers.plotting.geography.create_line_map import create_line_map
from mechaphlowers.plotting.geography.type_hints import (
    SupportGeoInfo,
)


class TestCreateLineMap:
    """Test suite for create_line_map function."""

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

    def test_create_line_map(self):
        """Test basic functionality of create_line_map."""
        fig = go.Figure()

        # Call the function
        create_line_map(fig, self.sample_supports)

        # Verify that traces were added to the figure
        assert len(fig.data) == 4  # 3 markers + 1 line trace

        # Verify marker traces
        marker_traces = [
            trace
            for trace in fig.data
            if trace.type == 'scattermapbox' and trace.mode == 'markers'
        ]
        assert len(marker_traces) == 3

        # Verify line trace
        line_traces = [
            trace
            for trace in fig.data
            if trace.type == 'scattermapbox' and trace.mode == 'lines'
        ]
        assert len(line_traces) == 1

        # Verify marker properties
        for i, trace in enumerate(marker_traces):
            assert np.allclose(
                trace.lat, [self.sample_supports[i]["gps"]["lat"]]
            )
            assert np.allclose(
                trace.lon, [self.sample_supports[i]["gps"]["lon"]]
            )
            assert trace.mode == 'markers'
            assert trace.marker.size == 10
            assert trace.marker.color == '#FF0000'
            assert trace.marker.opacity == 0.8
            assert trace.hoverinfo == "text"
            assert f"Support {i}" in trace.name

        # Verify line trace properties
        line_trace = line_traces[0]
        expected_lats = np.array(
            [support["gps"]["lat"] for support in self.sample_supports]
        )
        expected_lons = np.array(
            [support["gps"]["lon"] for support in self.sample_supports]
        )
        assert np.allclose(line_trace.lat, expected_lats)
        assert np.allclose(line_trace.lon, expected_lons)
        assert line_trace.mode == 'lines'
        assert line_trace.line.color == '#666666'
        assert line_trace.line.width == 2
        assert line_trace.hoverinfo == 'skip'
        assert line_trace.showlegend is False
