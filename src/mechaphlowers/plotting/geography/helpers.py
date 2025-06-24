import numpy as np

from mechaphlowers.data.geography.helpers import (
    bearing_to_direction,
    gps_to_bearing,
    gps_to_elevation,
    gps_to_lambert93,
    haversine,
)
from mechaphlowers.plotting.geography.type_hints import (
    GpsPoint,
    Lambert93Point,
    SupportGeoInfo,
)


def geo_info_from_gps(gps_points: list[GpsPoint]) -> list[SupportGeoInfo]:
    """
    Create a list of support geo info from a list of gps points.
    Args:
        gps_points: A list of gps points.
    Returns:
        A list of support geo info.
    """
    geo_infos: list[SupportGeoInfo] = []
    lats = np.array([point["lat"] for point in gps_points])
    lons = np.array([point["lon"] for point in gps_points])
    elevations = gps_to_elevation(lats, lons)
    lambert_93_tuples = gps_to_lambert93(lats, lons)
    distances = haversine(lats[:-1], lons[:-1], lats[1:], lons[1:])
    bearings = gps_to_bearing(lats[:-1], lons[:-1], lats[1:], lons[1:])
    directions = bearing_to_direction(bearings)

    for i in range(len(gps_points) - 1):
        distance_to_next = distances[i]
        bearing_to_next = bearings[i]
        direction_to_next = directions[i]
        elevation = elevations[i]
        geo_info = SupportGeoInfo(
            gps=gps_points[i],
            distance_to_next=distance_to_next,
            bearing_to_next=bearing_to_next,
            direction_to_next=direction_to_next,
            lambert_93=Lambert93Point(
                x=float(lambert_93_tuples[0][i]),
                y=float(lambert_93_tuples[1][i]),
            ),
            elevation=elevation,
        )
        geo_infos.append(geo_info)
    return geo_infos
