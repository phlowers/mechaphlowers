# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.entities.geography import (
    get_dist_and_angles_from_gps,
    get_gps_from_arrays,
)


def test_section_array_to_gps_0():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(["1", "2", "three", "4", "5"]),
                "suspension": np.array([False, True, True, True, False]),
                "conductor_attachment_altitude": np.array([20, 5, 10, 0, 0]),
                "crossarm_length": np.array([10, 12.1, 10, 10.1, 5]),
                "line_angle": np.array([90, 90, 90, 90, 90]),
                "insulator_length": np.array([0, 4, 3.2, 0, 0]),
                "span_length": np.array([500, 500, 500.0, 500.0, np.nan]),
                "insulator_mass": np.array(
                    [1000.0, 500.0, 500.0, 500.0, 1000.0]
                ),
            }
        )
    )
    section_array.add_units({"line_angle": "deg"})
    all_lats, all_lons = get_gps_from_arrays(
        start_lat=48.8566,
        start_lon=2.3522,
        azimuth=0,
        line_angles_degrees=-np.degrees(
            section_array.data.line_angle.to_numpy()
        ),
        span_length=section_array.data.span_length.to_numpy(),
    )

    # show_street_map(section_array, all_lats, all_lons)


def test_section_array_to_gps_1():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(
                    ["support 1", "2", "three", "support 4", "5"]
                ),
                "suspension": np.array([False, True, True, True, False]),
                "conductor_attachment_altitude": np.array(
                    [2.2, 5, -0.12, 0, 0]
                ),
                "crossarm_length": np.array([10, 12.1, 10, 10.1, 5]),
                "line_angle": np.array([0, 10, 15, 20, 30]),
                "insulator_length": np.array([0, 4, 3.2, 0, 0]),
                "span_length": np.array([300, 400, 500.0, 600.0, np.nan]),
                "insulator_mass": np.array(
                    [1000.0, 500.0, 500.0, 500.0, 1000.0]
                ),
            }
        )
    )
    section_array.add_units({"line_angle": "deg"})
    all_lats, all_lons = get_gps_from_arrays(
        start_lat=48.8566,
        start_lon=2.3522,
        azimuth=0,
        line_angles_degrees=-np.degrees(
            section_array.data.line_angle.to_numpy()
        ),
        span_length=section_array.data.span_length.to_numpy(),
    )

    # show_street_map(section_array, all_lats, all_lons)


def test_section_array_to_gps_2():
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": np.array(
                    ["support 1", "2", "three", "support 4", "5"]
                ),
                "suspension": np.array([False, True, True, True, False]),
                "conductor_attachment_altitude": np.array(
                    [2.2, 5, -0.12, 0, 0]
                ),
                "crossarm_length": np.array([10, 12.1, 10, 10.1, 5]),
                "line_angle": np.array([0, 10, -15, 20, 30]),
                "insulator_length": np.array([0, 4, 3.2, 0, 0]),
                "span_length": np.array([300, 400, 500.0, 600.0, np.nan]),
                "insulator_mass": np.array(
                    [1000.0, 500.0, 500.0, 500.0, 1000.0]
                ),
            }
        )
    )
    section_array.add_units({"line_angle": "deg"})
    all_lats, all_lons = get_gps_from_arrays(
        start_lat=48.8566,  # in rads: 0.852708
        start_lon=2.3522,  # in rads: 0.041053
        azimuth=np.pi / 2,
        line_angles_degrees=-np.degrees(
            section_array.data.line_angle.to_numpy()
        ),
        span_length=section_array.data.span_length.to_numpy(),
    )

    # show_street_map(section_array, all_lats, all_lons)


def test_gps_to_section_array_1():
    # Radius 6371 000
    lats = np.degrees(
        np.array(
            [0.852708, 0.8527550884, 0.852816919, 0.8528880459, 0.8529546364]
        )
    )
    lons = np.degrees(
        np.array(
            [0.041053, 0.041053, 0.0410695724, 0.0411199932, 0.0412212353]
        )
    )

    distances, angles = get_dist_and_angles_from_gps(lats, lons)
    np.testing.assert_allclose(
        distances, np.array([300, 400, 500, 600, np.nan]), atol=1e-3
    )
    np.testing.assert_allclose(angles, np.array([0, 10, 15, 20, 0]), atol=1e-3)


def test_gps_to_section_array_2():
    lats = np.degrees(
        np.array(
            [0.852708, 0.8527079987, 0.8526970941, 0.8527039307, 0.8526795512]
        )
    )
    lons = np.degrees(
        np.array(
            [0.041053, 0.0411245687, 0.0412185428, 0.0413373695, 0.0414756252]
        )
    )
    distances, angles = get_dist_and_angles_from_gps(lats, lons)
    np.testing.assert_allclose(
        distances, np.array([300, 400, 500, 600, np.nan]), atol=1e-3
    )
    np.testing.assert_allclose(
        angles, np.array([0, 10, -15, 20, 0]), atol=1e-3
    )


def test_round_trip():
    # --- Forward: section geometry → GPS ---
    # Put 0 at first and last support, because inverse operation can't manage them
    line_angles = np.array([0.0, 10.0, -15.0, 20.0, 0])
    span_lengths = np.array([300.0, 400.0, 500.0, 600.0, np.nan])

    lats, lons = get_gps_from_arrays(
        start_lat=48.8566,
        start_lon=2.3522,
        azimuth=90.0,  # first span heads East
        line_angles_degrees=line_angles,
        span_length=span_lengths,
    )

    # --- Inverse: GPS → section geometry ---
    recovered_distances, recovered_angles = get_dist_and_angles_from_gps(
        lats, lons
    )

    np.testing.assert_allclose(recovered_distances, span_lengths, atol=1e-3)
    np.testing.assert_allclose(recovered_angles, line_angles, atol=1e-3)


def show_street_map(
    section_array: SectionArray, all_lats: np.ndarray, all_lons: np.ndarray
):
    # Create hover text with support information
    hover_text = [f"Start: {section_array.data.name.iloc[0]}"] + [
        f"Support: {name}<br>"
        f"Span: {span:.1f}m<br>"
        f"Altitude: {alt:.2f}m<br>"
        f"Suspension: {susp}"
        for name, span, alt, susp in zip(
            section_array.data.name.iloc[1:],
            section_array.data.span_length.iloc[:-1],
            section_array.data.conductor_attachment_altitude.iloc[1:],
            section_array.data.suspension.iloc[1:],
        )
    ]

    # Create the map figure
    fig = go.Figure()

    # Add line connecting the supports
    fig.add_trace(
        go.Scattermapbox(
            lat=all_lats,
            lon=all_lons,
            mode='lines+markers',
            marker=dict(size=10, color='red'),
            line=dict(width=2, color='blue'),
            text=hover_text,
            hoverinfo='text',
            name='Section supports',
        )
    )

    # Calculate center point for map
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    # Update layout with mapbox
    fig.update_layout(
        mapbox={
            "style": "open-street-map",
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": 13,
        },
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=700,
        width=1000,
        showlegend=True,
    )

    fig.show(config={"scrollZoom": True})
