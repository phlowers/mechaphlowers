# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)

requests_installed = False

try:
    import requests

    requests_installed = True
except ImportError:
    pass

OPEN_ELEVATION_API_URL = "https://api.open-elevation.com/api/v1/lookup"


class IElevationService(ABC):
    """Abstract base class for elevation services.

    This interface defines the contract for services that provide elevation data
    based on GPS coordinates.
    """

    @abstractmethod
    def get_elevation(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> np.ndarray:
        """
        Fetch elevation data for a list of locations.

        Args:
            lat (np.ndarray): Latitude of the location in degrees
            lon (np.ndarray): Longitude of the location in degrees

        Returns:
            np.ndarray: Elevation in meters
        """
        pass


class OpenElevationService(IElevationService):
    """Implementation of elevation service using Open-Elevation API."""

    def get_elevation(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> np.ndarray:
        """
        Fetch elevation data for a list of locations using Open-Elevation API

        Args:
            lat (np.ndarray): Latitude of the location in degrees
            lon (np.ndarray): Longitude of the location in degrees

        Returns:
            np.ndarray: Elevation in meters
        """
        if not requests_installed:
            raise ImportError(
                "requests is not installed, use the full installation to use this service"
            )

        # Format locations for the API
        payload = {
            "locations": [
                {
                    "latitude": lat.tolist(),
                    "longitude": lon.tolist(),
                }
            ]
        }

        try:
            response = requests.post(OPEN_ELEVATION_API_URL, json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            return np.array(
                [result["elevation"] for result in data["results"]]
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching elevation data: {e}")
            return np.zeros(len(lat))  # Return zeros if request fails

    def __call__(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        return self.get_elevation(lat, lon)


gps_to_elevation = OpenElevationService()
