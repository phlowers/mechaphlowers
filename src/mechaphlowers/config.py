"""Module for mechaphlowers configuration settings"""

from dataclasses import dataclass


@dataclass
class GraphicsConfig:
	"""Graphics configuration class."""

	resolution: int = 7
	marker_size: float = 3.0


@dataclass
class SolverConfig:
	"""Solver configuration class."""

	sagtension_zeta: int = 10
	deformation_imag_thresh: float = 1e-5


@dataclass
class Config:
	"""Configuration class for mechaphlowers settings.

	This class is not intended to be used directly. Other classes
	are using the options instance to provide configuration settings. Default values are set in the class.
	`options` is available in the module mechaphlowers.config.

	Attributes:
		graphics_resolution (int): Resolution of the graphics.
		graphics_marker_size (float): Size of the markers in the graphics.
	"""

	def __init__(self):
		self._graphics = GraphicsConfig()
		self._solver = SolverConfig()

	@property
	def graphics(self) -> GraphicsConfig:
		"""Graphics configuration property."""
		return self._graphics

	@property
	def solver(self) -> SolverConfig:
		"""Solver configuration property."""
		return self._solver


# Declare below a ready to use options object
options = Config()
