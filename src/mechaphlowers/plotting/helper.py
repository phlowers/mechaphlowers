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