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

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the bearing between two points
    Returns bearing in degrees from north (0-360)
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    return bearing

def get_direction_name(bearing):
    """
    Convert bearing angle to cardinal direction name
    """
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = round(bearing / 45) % 8
    return directions[index]

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the markers
# markers = [
#     {'name': 'Paris', 'lat': 48.8566, 'lon': 2.3522, 'color': '#FF0000', 'size': 12},
#     {'name': 'Lyon', 'lat': 45.7640, 'lon': 4.8357, 'color': '#00FF00', 'size': 10},
#     {'name': 'Marseille', 'lat': 43.2965, 'lon': 5.3898, 'color': '#0000FF', 'size': 10}
# ]

# Create the map figure
def create_map(df):
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


def generate_section_array(size_section=15):
    # Generate a SectionArray
    name = []
    suspension = []
    altitude = []
    crossarm_length = []
    line_angle = []
    insulator_length = []
    span_length = []
    coordinates = []

    start_point = {'lat': 48.8566, 'lon': 2.3522}
    
    for i in range(size_section):
        name.append(f"Section {i}")
        if i == 0 or i == size_section-1:
            suspension.append(False)
            insulator_length.append(0.)
        else:
            suspension.append(True)
            insulator_length.append(float(np.random.uniform(5, 10)))
        altitude.append(np.random.uniform(20, 150))
        crossarm_length.append(4.0)
        
        current_span_length = float(np.random.uniform(80, 1000))
        if i == size_section-1:
            span_length.append(np.nan)
        else:
            span_length.append(current_span_length) 
        line_angle.append(np.random.uniform(0, 60))
        new_point = reverse_haversine(start_point['lat'], start_point['lon'], 180, current_span_length/1000)
        coordinates.append(new_point)
        start_point = new_point
            
    section_data = {
        "name": name,
        "suspension": suspension,
        "conductor_attachment_altitude": altitude,
        "crossarm_length": crossarm_length,
        "insulator_length": insulator_length,
        "span_length": span_length,
        "line_angle": line_angle,
        "coordinates": coordinates
    }
    return pd.DataFrame(section_data)

def add_obstacle(df, x,y,z,type_obstacle, name_obstacle, support):

    df.x.cumsum()
    new_df = pd.DataFrame({
        "x": [x],
        "y": [y],
        "z": [z],
        "type": [type_obstacle],
        "support": [support],
        "canton": ["obstacle"]
    })
    return pd.concat([df,new_df], ignore_index=True)



def create_map2(df, size_section, size_multiple):
    
    mph.options.graphics.resolution = res = 10

    section = SectionArray(data=df)

    # set sagging parameter and temperatur 
    section.sagging_parameter = 800
    section.sagging_temperature = 15

    # Provide section to SectionDataFrame
    frame = SectionDataFrame(section)
    cable_array = CableArray(
    pd.DataFrame(
        {
            "section": [345.55],
            "diameter": [22.4],
            "linear_weight": [9.55494],
            "young_modulus": [59],
            "dilatation_coefficient": [23],
            "temperature_reference": [15],
            "a0": [0],
            "a1": [59],
            "a2": [0],
            "a3": [0],
            "a4": [0],
            "b0": [0],
            "b1": [0],
            "b2": [0],
            "b3": [0],
            "b4": [0],
        }
    )
    )
    frame.add_cable(cable_array)


    weather = WeatherArray(
        pd.DataFrame(
            {
                "ice_thickness": [0]*size_section,
                "wind_pressure": [540.12, 0.0, 212.0, 53.0, 0.]*size_multiple,
            }
        )
    )
    frame.add_weather(weather=weather)

    # Display figure
    fig = go.Figure()
    frame.plot.line3d(fig, "full")
    fig.update_layout(
        width=1200,
        height=500,
        autosize=False,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            aspectmode='manual',
            aspectratio=dict(x=3, y=0.2, z=0.5),
            camera=dict(
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0.02, y=-2, z=0.2)
            )
        )
    )
    # json_3d = fig.to_json()

    return fig
    
    # def get_canton(frame, shift_arm, shift_altitude, phase_name):
    
    #     save1 = frame.section_array._data.conductor_attachment_altitude
    #     save2 = frame.section_array._data.crossarm_length
        
    #     frame.section_array._data.conductor_attachment_altitude += shift_altitude
    #     frame.section_array._data.crossarm_length += shift_arm
    #     frame.update()
    #     line_points = frame.get_coordinates()

    #     support_points = plt.plot.get_support_points(frame.data)

    #     insulator_points = plt.plot.get_insulator_points(frame.data)
        
    #     spans = pd.DataFrame(line_points, columns=["x", "y", "z"])
    #     spans["support"] = df.loc[0:size_section-2,"name"].repeat(res).values
    #     spans["type"] = "span"
        
    #     supports = pd.DataFrame(support_points, columns=["x", "y", "z"])
    #     supports["support"] = df.name.repeat(6).values
    #     supports["type"] = "support"
        
    #     insulators = pd.DataFrame(insulator_points, columns=["x", "y", "z"])
    #     insulators["support"] = df.name.repeat(3).values
    #     insulators["type"] = "insulator"
        
        
    #     pdf = pd.concat([spans,supports,insulators], ignore_index=True)
    #     pdf["canton"] = phase_name
        
    #     frame.section_array._data.conductor_attachment_altitude = save1
    #     frame.section_array._data.crossarm_length = save2
        
    #     return pdf
    
    # canton_0 = get_canton(frame, -4, 10, "garde")
    # canton_1 = get_canton(frame, 0, -10, "phase_1")
    # canton_2 = get_canton(frame, -8, -5, "phase_2")
    # canton_3 = get_canton(frame, 0, 0, "phase_3")
    
    # section_0 = "Section 0"
    # section_1 = "Section 1"
    # section_2 = "Section 2"

    # canton_1 = add_obstacle(canton_1, 150, 0, 20, "toit", "obstacle_1", section_0)
    # canton_1 = add_obstacle(canton_1, 150, 10, 30, "toit", "obstacle_1", section_0)
    # canton_1 = add_obstacle(canton_1, 150, 20, 40, "toit", "obstacle_1", section_0)

    # canton_1 = add_obstacle(canton_1, 700, -5, 30, "arbre", "obstacle_2", section_1)
    # canton_1 = add_obstacle(canton_1, 700, -5, 0, "arbre", "obstacle_2", section_1)

    # canton_1 = add_obstacle(canton_1, 800, 0, 0, "sol", "obstacle_2", section_2)
    # canton_1 = add_obstacle(canton_1, 900, 0, -30, "sol", "obstacle_2", section_2)
    # canton_1 = add_obstacle(canton_1, 1000, 0, -90, "sol", "obstacle_2", section_2)   
    # canton_1 = add_obstacle(canton_1, 1100, 0, -150, "sol", "obstacle_2", section_2)
    # canton_1 = add_obstacle(canton_1, 1200, 0, -200, "sol", "obstacle_2", section_2)
    # canton_1 = add_obstacle(canton_1, 1300, 0, -190, "sol", "obstacle_2", section_2)
    # canton_1 = add_obstacle(canton_1, 1400, 0, -90, "sol", "obstacle_2", section_2)
    # canton_1 = add_obstacle(canton_1, 1590, 0, 0, "sol", "obstacle_2", section_2)
    # lit = pd.concat([canton_0,canton_1,canton_2,canton_3], ignore_index=True)
    # lit["color_select"] = lit.canton + lit.type
    
    # # fig2
    
    # fig = px.line(canton_1, x="x", y="z", color="type",hover_data="support")

    # fig.update_traces(line=dict(width=5))

    # aspect_ratio = dict(x=1, y=0.05, z=0.5)

    # fig.update_layout(
    #         scene=dict(
    #             aspectratio=aspect_ratio,
    #             aspectmode="manual",
    #             camera=dict(
    #                 up=dict(x=0, y=0, z=1),
    #                 eye=dict(
    #                     x=0,
    #                     y=-1 / 1,
    #                     z=0,
    #                 ),
    #             ),
    #         )
    #     )
    # # fig.update_layout(
    # #     xaxis=dict(
    # #         rangeselector=dict(
    # #         ),
    # #         rangeslider=dict(
    # #             visible=True,
    # #             # thickness=.9
    # #         ),
    # #         type="linear"
    # #     )
    # # )
    # json_2d = fig.to_json()
    
    # fig = px.line(lit[lit.type.isin(["insulator", "support", ])], x="y", y="z", color="type",hover_data=["canton","support"], animation_frame="support", line_group="canton", height=1000, width=400)

    # fig.update_traces(line=dict(width=5))

    # aspect_ratio = dict(x=1, y=0.05, z=0.5)

    # fig.update_layout(
    #         scene=dict(
    #             aspectratio=aspect_ratio,
    #             aspectmode="data",
    #             camera=dict(
    #                 up=dict(x=0, y=1, z=1),
    #                 eye=dict(
    #                     x=0,
    #                     y=-1 / 1,
    #                     z=0,
    #                 ),
    #             ),
    #         )
    #     )
    
    # support_2d = fig.to_json()
    
    # fig = px.line_3d(canton_1, x="x", y="y", z="z", color="type", line_group="canton" ,
    #              hover_data=["support", "canton"], animation_frame="support", width=1200, height=800)

    # fig.update_traces(line=dict(width=8))

    # aspect_ratio = dict(x=1, y=0.05, z=0.5)
    # fig["layout"].pop("updatemenus") 
    # fig.update_layout(
    #         scene=dict(
    #             aspectratio=aspect_ratio,
    #             aspectmode="manual",
    #             camera=dict(
    #                 up=dict(x=0, y=0, z=1),
    #                 eye=dict(
    #                     x=0,
    #                     y=-1 / 1,
    #                     z=0,
    #                 ),
    #             ),
    #         )
    #     )

    # scene_aspect_ratio ="scene.aspectratio"
    # fig.update_layout(
    #     updatemenus=[
    #         dict(
    #             type = "buttons",
    #             direction = "left",
    #             buttons=list([
    #                 dict(
    #                     args=["scene.camera.eye", dict(x=0, y=-1, z=0)],
    #                     label="profil",
    #                     method="relayout"
    #                 ),
    #                 dict(
    #                     args=["scene.camera.eye", dict(x=-1/.9, y=0, z=0)],
    #                     label="contre profil",
    #                     method="relayout"
    #                 ),
    #                 dict(
    #                     args=[scene_aspect_ratio, dict(x=1, y=0.00005, z= .5)],
    #                     label="Mode 2D ligne",
    #                     method="relayout"
    #                 ),
    #                 dict(
    #                     args=[scene_aspect_ratio, dict(x=1e-5, y=1, z= .5)],
    #                     label="Mode 2D support",
    #                     method="relayout"
    #                 ),
    #                 dict(
    #                     args=[scene_aspect_ratio, dict(x=1, y=0.05, z= .5)],
    #                     label="Mode 3D support",
    #                     method="relayout"
    #                 ),
                    
    #             ]),
    #             pad={"r": 10, "t": 10},
    #             showactive=True,
    #             x=0.11,
    #             xanchor="left",
    #             y=1.1,
    #             yanchor="top"
    #         ),
    #     ]
        
    # )




    
    # json_section_3d = fig.to_json()

    # return {"lit": lit.to_dict()}



def gps_to_lambert93(lat, lon):
    """
    Convert GPS coordinates (latitude/longitude) to Lambert 93 coordinates
    Based on the official French coordinate system
    
    Parameters:
    lat: float - Latitude in decimal degrees
    lon: float - Longitude in decimal degrees
    
    Returns:
    tuple: (x, y) coordinates in Lambert 93 (in meters)
    """
    # Constants for Lambert 93
    a = 6378137.0  # Semi-major axis of the ellipsoid (in meters)
    e = 0.08181919106  # First eccentricity of the ellipsoid
    n = 0.7256077650  # Lambert 93 projection parameter
    c = 11754255.426  # Lambert 93 projection parameter
    xs = 700000.0  # X coordinate of the origin (in meters)
    ys = 12655612.050  # Y coordinate of the origin (in meters)
    lon0 = math.radians(3.0)  # Longitude of origin (in radians)
    lat0 = math.radians(46.5)  # Latitude of origin (in radians)
    
    # Convert input coordinates to radians
    lat = math.radians(lat)
    lon = math.radians(lon)
    
    # Calculate intermediate values
    e2 = e * e
    lat0_sin = math.sin(lat0)
    lat0_cos = math.cos(lat0)
    
    # Calculate the conformal latitude
    lat_sin = math.sin(lat)
    lat_cos = math.cos(lat)
    
    # Calculate the conformal latitude
    lat_c = 2 * math.atan(math.tan(math.pi/4 + lat/2) * 
                         math.pow((1 - e * lat_sin) / (1 + e * lat_sin), e/2)) - math.pi/2
    
    # Calculate the conformal latitude of the origin
    lat0_c = 2 * math.atan(math.tan(math.pi/4 + lat0/2) * 
                          math.pow((1 - e * lat0_sin) / (1 + e * lat0_sin), e/2)) - math.pi/2
    
    # Calculate the radius of the conformal sphere
    r = c * math.pow(math.tan(math.pi/4 + lat0_c/2), n)
    
    # Calculate the radius of the conformal sphere at the point
    r_p = c * math.pow(math.tan(math.pi/4 + lat_c/2), n)
    
    # Calculate the angle between the meridian of origin and the point
    theta = n * (lon - lon0)
    
    # Calculate the Lambert 93 coordinates
    x = xs + r_p * math.sin(theta)
    y = ys - r_p * math.cos(theta)
    
    return x, y

def create_map3(df, size_section, size_multiple):
    
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

def main():
        # Define the app layout
    np.random.seed(142)
    # np.random.seed(42)
    size_multiple = 5
    size_section = size_multiple*5
    df = generate_section_array(size_section)
    app.layout = html.Div([
        # dcc.Graph(
        #     id='map',
        #     figure=create_map2(df, size_section, size_multiple),
        #     config={
        #         'displayModeBar': True,
        #         'displaylogo': False,
        #         'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d']
        #     }
        # ),
        dcc.Graph(
            id='map2',
            figure=create_map(df),
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d']
            }
        ),
        dcc.Graph(
            id='map3',
            figure=create_map3(df, size_section, size_multiple),
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'select2d', 'lasso2d', 'autoScale2d']
            }
        )
    ], style={'width': '100%', 'height': '100%', 'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'center', 'alignItems': 'center'})

    app.run(debug=True)

# result = main()
if __name__ == '__main__':
    main()