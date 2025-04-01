"""Module for mechaphlowers configuration settings"""

from dataclasses import dataclass


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

	graphics_resolution: int = 7
	graphics_marker_size: float = 3.0


# Declare below a ready to use options object
options = Config()
