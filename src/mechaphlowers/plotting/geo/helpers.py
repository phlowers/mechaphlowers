import math

def reverse_haversine(lat, lon, bearing, distance):
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
    
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    angdist = distance / R
    theta = math.radians(bearing)
    
    lat2 = math.degrees(
        math.asin(
            math.sin(lat1) * math.cos(angdist) + 
            math.cos(lat1) * math.sin(angdist) * math.cos(theta)
        )
    )
    
    lon2 = math.degrees(
        lon1 + math.atan2(
            math.sin(theta) * math.sin(angdist) * math.cos(lat1),
            math.cos(angdist) - math.sin(lat1) * math.sin(math.radians(lat2))
        )
    )
    
    return {'lat': lat2, 'lon': lon2}

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

def parse_coordinates(coordinates):
    """
    Parse coordinates in the form:
    coords = array([[   0.        ,    0.        ,    0.        ],
       [   0.        ,    0.        ,   30.        ],
       [   0.        ,   40.        ,   30.        ],
       [          nan,           nan,           nan],
       [ 500.        ,    0.        ,    0.        ],
       [ 500.        ,    0.        ,   40.        ],
       [ 507.65366865,   18.47759065,   40.        ],
       [          nan,           nan,           nan],
       [ 825.26911935, -325.26911935,    0.        ],
       [ 825.26911935, -325.26911935,   60.        ],
       [ 817.50454799, -354.24689413,   60.        ],
       [          nan,           nan,           nan],
       [1327.55054902, -190.68321589,    0.        ],
       [1327.55054902, -190.68321589,   70.        ],
       [1327.55054902, -240.68321589,   70.        ],
       [          nan,           nan,           nan]])
    """
    splitted_coords = np.split(coordinates, 4)
    support_bases = [x[0] for x in splitted_coords]
    

def calculate_gps_from_distances(lat_a, lon_a, x_meters, y_meters, z_meters):
    """
    Calculate GPS coordinates of point B given point A's coordinates and x,y,z distances in meters.
    
    Args:
        lat_a (float): Latitude of point A in degrees
        lon_a (float): Longitude of point A in degrees
        x_meters (float): Distance from west to east in meters (positive = east, negative = west)
        y_meters (float): Distance from south to north in meters (positive = north, negative = south)
        z_meters (float): Distance from bottom to top in meters (positive = up, negative = down)
        
    Returns:
        dict: Dictionary with 'lat' and 'lon' keys containing the GPS coordinates of point B
    """
    # Earth's radius in meters
    R = 6378137.0
    
    # Convert distances to degrees
    # 1 degree of latitude is approximately 111,111 meters
    lat_change = y_meters / 111111.0
    
    # 1 degree of longitude varies with latitude
    # At the equator, 1 degree is about 111,111 meters
    # At other latitudes, multiply by cos(latitude)
    lon_change = x_meters / (111111.0 * math.cos(math.radians(lat_a)))
    
    # Calculate new coordinates
    lat_b = lat_a + lat_change
    lon_b = lon_a + lon_change
    
    # Note: z_meters is not used in the calculation as GPS coordinates are 2D
    # The z coordinate would be used for altitude, but standard GPS coordinates
    # don't include this information
    
    return {'lat': lat_b, 'lon': lon_b}



