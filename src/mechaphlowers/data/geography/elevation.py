import numpy as np
import requests

OPEN_ELEVATION_API_URL = "https://api.open-elevation.com/api/v1/lookup"


def gps_to_elevation(
    lat: np.typing.NDArray[np.float64], lon: np.typing.NDArray[np.float64]
) -> np.typing.NDArray[np.float64]:
    """
    Fetch elevation data for a list of locations using Open-Elevation API

    Args:
        lat (np.typing.NDArray[np.float64]): Latitude of the location in degrees
        lon (np.typing.NDArray[np.float64]): Longitude of the location in degrees

    Returns:
        np.typing.NDArray[np.float64]: Elevation in meters
    """

    # Format locations for the API
    payload = {
        "locations": [
            {
                "latitude": lat,
                "longitude": lon,
            }
        ]
    }

    try:
        response = requests.post(OPEN_ELEVATION_API_URL, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        return np.array([result["elevation"] for result in data["results"]])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching elevation data: {e}")
        return np.zeros(len(lat))  # Return zeros if request fails
