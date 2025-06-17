import dash
from dash import html, dcc
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
from mechaphlowers import SectionDataFrame
from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.entities.arrays import CableArray, WeatherArray
import mechaphlowers as mph
from mechaphlowers import plotting as plt
from mechaphlowers.plotting.helper import reverse_haversine
import json
import os
import math

# Initialize the Dash app
app = dash.Dash(__name__)


# Create the map figure
def create_line_map(df):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "./reseau-aerien-haute-tension-htb-corse-with-elevation.json"), "r") as f:
        data = json.load(f)
        markers = data
    
    # Create marker traces
    marker_traces = []
    lambert_coords = []  # Store Lambert 93 coordinates


    for i, marker in enumerate(markers):
        # Calculate distance and bearing to next point if it exists
        distance_info = ""
        bearing_info = ""
        if i < len(markers) - 1:
            next_marker = markers[i + 1]
            distance = haversine_distance(
                marker['geo_point_2d']['lat'],
                marker['geo_point_2d']['lon'],
                next_marker['geo_point_2d']['lat'],
                next_marker['geo_point_2d']['lon']
            )
            bearing = calculate_bearing(
                marker['geo_point_2d']['lat'],
                marker['geo_point_2d']['lon'],
                next_marker['geo_point_2d']['lat'],
                next_marker['geo_point_2d']['lon']
            )
            direction = get_direction_name(bearing)
            distance_info = f"<br>Distance to next: {distance:.2f} km"
            bearing_info = f"<br>Bearing to next: {bearing:.1f}° ({direction})"
            elevation_info = f"<br>Elevation: {marker['elevation']:.2f} m"
            x, y = gps_to_lambert93(
                marker['geo_point_2d']['lat'],
                marker['geo_point_2d']['lon']
            )
            lambert_coords.append((x, y))


        marker_traces.append(
            go.Scattermapbox(
                lat=[marker['geo_point_2d']['lat']],
                lon=[marker['geo_point_2d']['lon']],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=10,
                    color='#FF0000',
                    opacity=0.8
                ),
                text=[i],
                hoverinfo="text",
                hovertemplate='<b>%{text}</b><br>' + 
                            'Lat: %{lat:.6f}<br>' + 
                            'Lon: %{lon:.6f}' +
                            distance_info +
                            bearing_info + 
                            elevation_info +
                            f"<br>Lambert X: {x:.2f}<br>Lambert Y: {y:.2f}"
                            '<extra></extra>',
                name=f"{marker['nom_ligne']} - {marker['geo_point_2d']['lat']}-{marker['geo_point_2d']['lon']}"
            )
        )

    # Create line trace connecting markers
    line_trace = go.Scattermapbox(
        lat=[marker['geo_point_2d']['lat'] for marker in markers],
        lon=[marker['geo_point_2d']['lon'] for marker in markers],
        mode='lines',
        line=dict(
            color='#666666',
            width=2
        ),
        hoverinfo='skip',
        showlegend=False
    )

    # Combine all traces
    traces = marker_traces + [line_trace]

    # Create the figure
    fig = go.Figure(data=traces)

    # Update the layout
    fig.update_layout(
        mapbox=dict(
            style='carto-positron',
            center=dict(
                lat=42.19476145,
                lon=8.80258205
            ),
            zoom=11
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        height=700,
        width=1200,
        showlegend=False
    )

    return fig



import plotly.express as px




def create_elevation_profile(df, size_section, size_multiple):
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "./reseau-aerien-haute-tension-htb-corse-with-elevation.json"), "r") as f:
        data = json.load(f)
        markers = data
    
    # Calculate cumulative distances and prepare elevation data
    cumulative_distance = [0]  # Start at 0
    elevations = [markers[0]['elevation']]  # First elevation
    lambert_coords = []  # Store Lambert 93 coordinates
    
    for i in range(len(markers)):
        # Convert GPS to Lambert 93
        x, y = gps_to_lambert93(
            markers[i]['geo_point_2d']['lat'],
            markers[i]['geo_point_2d']['lon']
        )
        lambert_coords.append((x, y))
        
        if i > 0:
            # Calculate distance to previous point
            distance = haversine_distance(
                markers[i-1]['geo_point_2d']['lat'],
                markers[i-1]['geo_point_2d']['lon'],
                markers[i]['geo_point_2d']['lat'],
                markers[i]['geo_point_2d']['lon']
            )
            # Add to cumulative distance
            cumulative_distance.append(cumulative_distance[-1] + distance)
            # Add elevation
            elevations.append(markers[i]['elevation'])

    # Create the figure
    fig = go.Figure()
    
    # Add the main line plot
    fig.add_trace(go.Scatter(
        x=cumulative_distance,
        y=elevations,
        mode='lines+markers',
        name='Elevation Profile',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=8, color='#1f77b4'),
        hovertemplate='Marker: %{customdata[0]}<br>Distance: %{x:.2f} km<br>Elevation: %{y:.2f} m<br>Lambert X: %{customdata[1][0]:.2f}<br>Lambert Y: %{customdata[1][1]:.2f}<extra></extra>',
        customdata=list(zip(range(len(markers)), lambert_coords)),  # Add marker indices as custom data
        # customdata2=lambert_coords  # Add Lambert coordinates as custom data
    ))

    # Update layout
    fig.update_layout(
        title='Elevation Profile of Power Line',
        xaxis_title='Cumulative Distance (km)',
        yaxis_title='Elevation (m)',
        width=1200,
        height=500,
        showlegend=False,
        hovermode='closest',
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGray'
        )
    )

    return fig
