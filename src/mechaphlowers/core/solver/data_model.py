from dataclasses import dataclass

import numpy as np

from mechaphlowers.entities.arrays import CableArray, SectionArray, WeatherArray

@dataclass
class DataContainer:
	wa: WeatherArray
	ca: CableArray
	sa: SectionArray
    
	@property
	def to_dict(self):
		return self.wa.to_numpy() | self.ca.to_numpy() | self.sa.to_numpy()

@dataclass
class Args:
	# def __init__(self, d):
	# 	self.span_length = x_n
	# 	self.elevation_difference = p
	span_length: np.ndarray
	elevation_difference: np.ndarray
	sagging_parameter: np.ndarray
	sagging_temperature: np.ndarray
	linear_weight: np.ndarray
	young_modulus: np.ndarray
	section: np.ndarray
	alpha: np.ndarray
	dilatation_coefficient: np.ndarray
	a0: np.ndarray
	a1: np.ndarray
	a2: np.ndarray
	a3: np.ndarray
	a4: np.ndarray

	L_ref: np.ndarray

	temp_ref: np.ndarray
	ice_thickness: np.ndarray
	diameter: np.ndarray

	weather: np.ndarray

		
	def __getitem__(self, key):
		return getattr(self, key)