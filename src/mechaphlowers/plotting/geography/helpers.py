import numpy as np

from mechaphlowers.data.geography.elevation import gps_to_elevation
from mechaphlowers.data.geography.helpers import (
    bearing_to_direction,
    gps_to_bearing,
    gps_to_lambert93,
    haversine,
)
from mechaphlowers.plotting.geography.entities import (
    SupportGeoInfo,
)


def geo_info_from_gps(
    lats: np.typing.NDArray[np.float64], lons: np.typing.NDArray[np.float64]
) -> SupportGeoInfo:
    """
    Create a list of support geo info from a list of gps points.
    Args:
        gps_points: A list of gps points.
    Returns:
        A list of support geo info.
    """
    elevations = gps_to_elevation(lats, lons)
    lambert_93_tuples = gps_to_lambert93(lats, lons)
    distances = haversine(lats[:-1], lons[:-1], lats[1:], lons[1:])
    bearings = gps_to_bearing(lats[:-1], lons[:-1], lats[1:], lons[1:])
    directions = bearing_to_direction(bearings)

    return SupportGeoInfo(
        gps=(lats, lons),
        elevation=elevations,
        distance_to_next=distances,
        bearing_to_next=bearings,
        direction_to_next=directions,
        lambert_93=lambert_93_tuples,
    )
