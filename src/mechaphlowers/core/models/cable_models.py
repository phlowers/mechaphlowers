from abc import ABC, abstractmethod

import numpy as np


class GeometricCableModel(ABC):
    """This abstract class is a base class for various models describing the cable in its own frame."""

    def __init__(
        self, span_length: float, elevation_difference: float, p: float
    ) -> None:
        self.span_length = span_length
        self.elevation_difference = elevation_difference
        self.p = p

    @abstractmethod
    def z(self, x: np.ndarray) -> np.ndarray:
        """Altitude of cable points depending on the abscissa, in the cable frame.

        Args:
            x: abscissa

        Returns:
            altitudes based on the sag tension parameter "p" stored in the model.
        """


class CatenaryCableModel(GeometricCableModel):
    def z(self, x: np.ndarray) -> np.ndarray:
        """Altitude of cable points in the cable frame, according to the catenary model."""
        return self.p * (np.cosh(x / self.p) - 1)
