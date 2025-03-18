import numpy as np
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.entities.arrays import (
	CableArray,
	SectionArray,
	WeatherArray,
)


class DataContainer:
	def __init__(self) -> None:  # keep this for mypy?
		self.name: np.ndarray
		self.suspension: np.ndarray
		self.conductor_attachment_altitude: np.ndarray
		self.crossarm_length: np.ndarray
		self.line_angle: np.ndarray
		self.insulator_length: np.ndarray
		self.span_length: np.ndarray
		self.elevation_difference: np.ndarray
		self.sagging_parameter: np.ndarray
		self.sagging_temperature: np.ndarray

		self.section: np.ndarray
		self.diameter: np.ndarray
		self.linear_weight: np.ndarray
		self.young_modulus: np.ndarray
		self.dilatation_coefficient: np.ndarray
		self.temperature_reference: np.ndarray
		self.a0: np.ndarray
		self.a1: np.ndarray
		self.a2: np.ndarray
		self.a3: np.ndarray
		self.a4: np.ndarray
		self.b0: np.ndarray | None  # Correct solution??? Or always 0?
		self.b1: np.ndarray | None
		self.b2: np.ndarray | None
		self.b3: np.ndarray | None
		self.b4: np.ndarray | None

		self.ice_thickness: np.ndarray
		self.wind_pressure: np.ndarray

	@property
	def stress_strain_polynomial_conductor(self) -> Poly:
		"""Converts coefficients in the dataframe into polynomial"""
		coefs_poly = np.array(
			[self.a0[0], self.a1[0], self.a2[0], self.a3[0], self.a4[0]]
		)
		return Poly(coefs_poly)

	def add_data_section(
		self,
		name: np.ndarray,
		suspension: np.ndarray,
		conductor_attachment_altitude: np.ndarray,
		crossarm_length: np.ndarray,
		line_angle: np.ndarray,
		insulator_length: np.ndarray,
		span_length: np.ndarray,
		elevation_difference: np.ndarray,
		sagging_parameter: np.ndarray,
		sagging_temperature: np.ndarray,
		**kwargs,  # keep kwargs?
	) -> None:
		self.name = name
		self.suspension = suspension
		self.conductor_attachment_altitude = conductor_attachment_altitude
		self.crossarm_length = crossarm_length
		self.line_angle = line_angle
		self.insulator_length = insulator_length
		self.span_length = span_length
		self.elevation_difference = elevation_difference
		self.sagging_parameter = sagging_parameter
		self.sagging_temperature = sagging_temperature

	def add_data_cable(
		self,
		section: np.ndarray,
		diameter: np.ndarray,
		linear_weight: np.ndarray,
		young_modulus: np.ndarray,
		dilatation_coefficient: np.ndarray,
		temperature_reference: np.ndarray,
		a0: np.ndarray,
		a1: np.ndarray,
		a2: np.ndarray,
		a3: np.ndarray,
		a4: np.ndarray,
		b0: np.ndarray | None = None,
		b1: np.ndarray | None = None,
		b2: np.ndarray | None = None,
		b3: np.ndarray | None = None,
		b4: np.ndarray | None = None,
		**kwargs,
	) -> None:
		self.section = section
		self.diameter = diameter
		self.linear_weight = linear_weight
		self.young_modulus = young_modulus
		self.dilatation_coefficient = dilatation_coefficient
		self.temperature_reference = temperature_reference
		self.a0 = a0
		self.a1 = a1
		self.a2 = a2
		self.a3 = a3
		self.a4 = a4
		self.b0 = b0
		self.b1 = b1
		self.b2 = b2
		self.b3 = b3
		self.b4 = b4

	def add_data_weather(
		self, ice_thickness: np.ndarray, wind_pressure: np.ndarray, **kwargs
	) -> None:
		self.ice_thickness = ice_thickness
		self.wind_pressure = wind_pressure

	def __getitem__(self, key):
		return getattr(self, key)


def factory_data_container(
	section_array: SectionArray,
	cable_array: CableArray,
	weather_array: WeatherArray,
) -> DataContainer:
	data_container = DataContainer()
	# data_container.add_section(
	# 	section_array.data["name"].to_numpy(),
	# 	section_array.data["suspension"].to_numpy(),
	# 	section_array.data["conductor_attachment_altitude"].to_numpy(),
	# 	section_array.data["crossarm_length"].to_numpy(),
	# 	section_array.data["line_angle"].to_numpy(),
	# 	section_array.data["insulator_length"].to_numpy(),
	# 	section_array.data["span_length"].to_numpy(),
	# 	section_array.data["elevation_difference"].to_numpy(),
	# 	section_array.data["sagging_parameter"].to_numpy(),
	# 	section_array.data["sagging_temperature"].to_numpy(),
	# )
	data_container.add_data_section(**(section_array.to_dict_with_numpy()))
	data_container.add_data_cable(**(cable_array.to_dict_with_numpy()))
	data_container.add_data_weather(**(weather_array.to_dict_with_numpy()))
	# Create .to_dict_numpy() methods in arrays?
	return data_container
