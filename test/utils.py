# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from mechaphlowers.entities.arrays import SectionArray


def generate_html_from_json(json_path: Path):
    """Generate HTML report from JSON benchmark results with improved formatting."""
    current_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(json_path) as f:
        data = json.load(f)

    # Count passed/failed
    passed = sum(1 for t in data["tests"] if t.get("status") == "passed")
    failed = sum(1 for t in data["tests"] if t.get("status") == "failed")
    total = len(data["tests"])

    html = f"""
    <html>
    <head>
        <title>Functional Benchmark Report</title>
        <style>
            body {{ font-family: Arial; margin: 20px; background-color: #f5f5f5; }}
            .summary {{ background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .passed {{ color: green; font-weight: bold; }}
            .failed {{ color: red; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .details {{ background-color: #f0f0f0; padding: 15px; border-radius: 3px; font-family: monospace; font-size: 11px; white-space: pre-wrap; word-wrap: break-word; line-height: 1.5; }}
            .error-details {{ background-color: #ffebee; padding: 15px; border-left: 4px solid #d32f2f; border-radius: 3px; font-family: monospace; font-size: 11px; white-space: pre-wrap; word-wrap: break-word; line-height: 1.5; color: #c62828; }}
        </style>
    </head>
    <body>
        <h1>Functional Benchmark Report</h1>
        <p><strong>Generated:</strong> {data['timestamp']}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total Tests: <strong>{total}</strong></p>
            <p><span class="passed">✓ Passed: {passed}</span></p>
            <p><span class="failed">✗ Failed: {failed}</span></p>
        </div>
        
        <table>
            <tr>
                <th>Test Name</th>
                <th>Status</th>
                <th>Details</th>
            </tr>
    """

    for test in data["tests"]:
        test_name = test.get('name', 'Unknown')
        status = test.get('status', 'unknown')
        status_class = (
            "passed"
            if status == "passed"
            else "failed"
            if status == "failed"
            else ""
        )

        # Extract all fields except name and status for details
        details_dict = {
            k: v for k, v in test.items() if k not in ['name', 'status']
        }

        # Format details as plain text
        details_text = ""
        for key, value in details_dict.items():
            # Convert snake_case to Title Case
            section_title = key.replace('_', ' ').title()
            details_text += f"{section_title}:\n{value}\n\n"

        # Choose styling based on presence of error
        details_class = (
            "error-details" if "error" in details_dict else "details"
        )
        details_html = (
            f'<div class="{details_class}">{details_text.strip()}</div>'
        )

        html += f"""
            <tr>
                <td><strong>{test_name}</strong></td>
                <td><span class="{status_class}">{status.upper()}</span></td>
                <td>{details_html}</td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    report_path = json_path.parent / f"{current_dt}_benchmark_report.html"
    with open(report_path, "w") as f:
        f.write(html)
    print(f"✓ HTML report saved to {report_path}")


def show_street_map(
    section_array: SectionArray, all_lats: np.ndarray, all_lons: np.ndarray
):
    """Displays a street map with points with their GPS coordinates, using Plotly.

    Args:
        section_array (SectionArray): The section array containing support data.
        all_lats (np.ndarray): Array of latitudes for the supports.
        all_lons (np.ndarray): Array of longitudes for the supports.
    """
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
