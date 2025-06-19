from typing import Dict, List

import numpy as np
import requests


def gps_to_elevation(locations: List[Dict[str, float]]) -> List[float]:
    """
    Fetch elevation data for a list of locations using Open-Elevation API
    """
    url = "https://api.open-elevation.com/api/v1/lookup"
    
    # Format locations for the API
    payload = {
        "locations": [{"latitude": loc["lat"], "longitude": loc["lon"]} for loc in locations]
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        return [result["elevation"] for result in data["results"]]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching elevation data: {e}")
        return [0.0] * len(locations)  # Return zeros if request fails
    


def reverse_haversine(lat: float, lon: float, bearing: float, distance: float) -> dict[str, float]:
    """
    Implementation of the reverse of Haversine formula. Takes one set of 
    latitude/longitude as a start point, a bearing, and a distance, and 
    returns the resultant lat/long pair.
    
    Args:
        lat (float): Starting latitude in degrees
        lon (float): Starting longitude in degrees  
        bearing (float): Bearing in degrees
        distance (float): Distance in kilometers
        
    Returns:
        dict: Dictionary with 'lat' and 'lon' keys containing the result coordinates
    """
    R = 6378.137  # Radius of Earth in km
    
    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    angdist = distance / R
    theta = np.radians(bearing)
    
    lat2 = np.degrees(
        np.arcsin(
            np.sin(lat1) * np.cos(angdist) + 
            np.cos(lat1) * np.sin(angdist) * np.cos(theta)
        )
    )
    
    lon2 = np.degrees(
        lon1 + np.arctan2(
            np.sin(theta) * np.sin(angdist) * np.cos(lat1),
            np.cos(angdist) - np.sin(lat1) * np.sin(np.radians(lat2))
        )
    )
    
    return {'lat': lat2, 'lon': lon2}

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def gps_to_lambert93(latitude: float, longitude: float) -> tuple[float, float]:
    """
    Convert GPS coordinates (WGS84) to Lambert 93 coordinates.
    
    Args:
        latitude (float): Latitude in decimal degrees (WGS84)
        longitude (float): Longitude in decimal degrees (WGS84)
    
    Returns:
        tuple: (X, Y) coordinates in Lambert 93 projection (in meters)
    """
    
    # Lambert 93 projection parameters
    # Semi-major axis of the GRS80 ellipsoid
    a = 6378137.0
    # Flattening of the GRS80 ellipsoid
    f = 1/298.257222101
    
    # Derived parameters
    e2 = 2*f - f*f  # First eccentricity squared
    
    # Lambert 93 specific parameters
    phi0 = np.radians(46.5)  # Latitude of origin
    phi1 = np.radians(44.0)  # First standard parallel
    phi2 = np.radians(49.0)  # Second standard parallel
    lambda0 = np.radians(3.0)  # Central meridian
    X0 = 700000.0  # False easting
    Y0 = 6600000.0  # False northing
    
    # Convert input coordinates to radians
    phi = np.radians(latitude)
    lambda_deg = np.radians(longitude)
    
    # Calculate auxiliary functions
    def m(phi):
        return np.cos(phi) / np.sqrt(1 - e2 * np.sin(phi)**2)
    
    def t(phi):
        return np.tan(np.pi/4 - phi/2) / ((1 - np.sqrt(e2) * np.sin(phi)) / (1 + np.sqrt(e2) * np.sin(phi)))**(np.sqrt(e2)/2)
    
    # Calculate projection constants
    m1 = m(phi1)
    m2 = m(phi2)
    
    t0 = t(phi0)
    t1 = t(phi1)
    t2 = t(phi2)
    
    # Calculate n and F
    n = (np.log(m1) - np.log(m2)) / (np.log(t1) - np.log(t2))
    F = m1 / (n * t1**n)
    
    # Calculate rho0 (radius at origin)
    rho0 = a * F * t0**n
    
    # Calculate rho and theta for the point
    t_phi = t(phi)
    rho = a * F * t_phi**n
    theta = n * (lambda_deg - lambda0)
    
    # Calculate Lambert 93 coordinates
    X = X0 + rho * np.sin(theta)
    Y = Y0 + rho0 - rho * np.cos(theta)
    
    return (X, Y)


def lambert93_to_gps(lambert_e: float, lambert_n: float) -> tuple[float, float]:
    """
    Convert Lambert 93 coordinates to WGS84 (longitude, latitude)
    
    Args:
        lambert_e (float): Lambert 93 Easting coordinate
        lambert_n (float): Lambert 93 Northing coordinate
    
    Returns:
        dict: Dictionary containing longitude and latitude in decimal degrees
    """
    
    constantes = {
        'GRS80E': 0.081819191042816,
        'LONG_0': 3,
        'XS': 700000,
        'YS': 12655612.0499,
        'n': 0.7256077650532670,
        'C': 11754255.4261
    }

    del_x = lambert_e - constantes['XS']
    del_y = lambert_n - constantes['YS']
    gamma = np.arctan(-del_x / del_y)
    r = np.sqrt(del_x * del_x + del_y * del_y)
    latiso = np.log(constantes['C'] / r) / constantes['n']
    
    # Iterative calculation for sinPhiit
    sin_phi_it0 = np.tanh(latiso + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * np.sin(1)))
    sin_phi_it1 = np.tanh(latiso + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * sin_phi_it0))
    sin_phi_it2 = np.tanh(latiso + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * sin_phi_it1))
    sin_phi_it3 = np.tanh(latiso + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * sin_phi_it2))
    sin_phi_it4 = np.tanh(latiso + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * sin_phi_it3))
    sin_phi_it5 = np.tanh(latiso + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * sin_phi_it4))
    sin_phi_it6 = np.tanh(latiso + constantes['GRS80E'] * np.arctanh(constantes['GRS80E'] * sin_phi_it5))

    long_rad = np.arcsin(sin_phi_it6)
    lat_rad = gamma / constantes['n'] + constantes['LONG_0'] / 180 * np.pi

    longitude = lat_rad / np.pi * 180
    latitude = long_rad / np.pi * 180

    return (latitude, longitude)


def gps_to_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the bearing between two points
    Returns bearing in degrees from north (0-360)
    """
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.arctan2(y, x)
    bearing = np.degrees(bearing)
    bearing = (bearing + 360) % 360
    return bearing

def bearing_to_direction(bearing: float) -> str:
    """
    Convert bearing angle to cardinal direction name
    """
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = np.round(bearing / 45) % 8
    return directions[int(index)]

def parse_support_distances_to_calculate_gps(coordinates: np.ndarray, first_support_gps: np.ndarray) -> np.ndarray:
    """
    Parse coordinates in the form:
    coords = array([[   0.        ,    0.        ,    0.        ],
       [   0.        ,    0.        ,   30.        ],
       [   0.        ,   40.        ,   30.        ],
       [          nan,           nan,           nan]])
    """
    splitted_coords = np.split(coordinates, 4)
    support_bases = [x[0] for x in splitted_coords]
    gps_coordinates = np.zeros((len(support_bases), 2))
    gps_coordinates[0] = first_support_gps
    for i, support_base in enumerate(support_bases):
        if i > 0:
            gps_coordinates[i] = distances_to_gps(gps_coordinates[i-1][0], gps_coordinates[i-1][1], support_base[0], support_base[1])
    return gps_coordinates

def distances_to_gps(lat_a: float, lon_a: float, x_meters: float, y_meters: float) -> tuple[float, float]:
    """
    Calculate GPS coordinates of point B given point A's coordinates and x,y,z distances in meters.
    
    Args:
        lat_a (float): Latitude of point A in degrees
        lon_a (float): Longitude of point A in degrees
        x_meters (float): Distance from west to east in meters (positive = east, negative = west)
        y_meters (float): Distance from south to north in meters (positive = north, negative = south)
        
    Returns:
        dict: Dictionary with 'lat' and 'lon' keys containing the GPS coordinates of point B
    """
    # Convert distances to degrees
    # 1 degree of latitude is approximately 111,111 meters
    lat_change = y_meters / 111111.0
    
    # 1 degree of longitude varies with latitude
    # At the equator, 1 degree is about 111,111 meters
    # At other latitudes, multiply by cos(latitude)
    lon_change = x_meters / (111111.0 * np.cos(np.radians(lat_a)))
    
    # Calculate new coordinates
    lat_b = lat_a + lat_change
    lon_b = lon_a + lon_change
    
    return lat_b, lon_b
