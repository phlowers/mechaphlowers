import numpy as np
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.entities.arrays import (
	CableArray,
	SectionArray,
	WeatherArray,
)


class DataContainer:
	def __init__(self) -> None:  # keep this for mypy?
		self.support_name: np.ndarray
		self.suspension: np.ndarray
		self.conductor_attachment_altitude: np.ndarray
		self.crossarm_length: np.ndarray
		self.line_angle: np.ndarray
		self.insulator_length: np.ndarray
		self.span_length: np.ndarray
		self.elevation_difference: np.ndarray
		self.sagging_parameter: np.ndarray
		self.sagging_temperature: np.ndarray

		self.cable_section_area: np.float64
		self.diameter: np.float64
		self.linear_weight: np.float64
		self.young_modulus: np.float64
		self.dilatation_coefficient: np.float64
		self.temperature_reference: np.float64

		self.polynomial_conductor: Poly
		self.polynomial_heart: Poly

		self.ice_thickness: np.ndarray
		self.wind_pressure: np.ndarray

	def add_section_array(self, section_array: SectionArray) -> None:
		self.support_name = section_array.data.name.to_numpy()
		self.suspension = section_array.data.suspension.to_numpy()
		self.conductor_attachment_altitude = (
			section_array.data.conductor_attachment_altitude.to_numpy()
		)
		self.crossarm_length = section_array.data.crossarm_length.to_numpy()
		self.line_angle = section_array.data.line_angle.to_numpy()
		self.insulator_length = section_array.data.insulator_length.to_numpy()
		self.span_length = section_array.data.span_length.to_numpy()
		self.elevation_difference = (
			section_array.data.elevation_difference.to_numpy()
		)
		self.sagging_parameter = (
			section_array.data.sagging_parameter.to_numpy()
		)
		self.sagging_temperature = (
			section_array.data.sagging_temperature.to_numpy()
		)

	def add_cable_array(self, cable_array: CableArray) -> None:
		if len(cable_array.data.section) != 1:
			raise NotImplementedError("CableArray should only contain one row")
		# TODO: use dict instead of [0]
		self.cable_section_area = cable_array.data.section[0]
		self.diameter = cable_array.data.diameter[0]
		self.linear_weight = cable_array.data.linear_weight[0]
		self.young_modulus = cable_array.data.young_modulus[0]
		self.dilatation_coefficient = cable_array.data.dilatation_coefficient[
			0
		]
		self.temperature_reference = cable_array.data.temperature_reference[0]

		self.polynomial_conductor = Poly(
			[
				cable_array.data.a0[0],
				cable_array.data.a1[0],
				cable_array.data.a2[0],
				cable_array.data.a3[0],
				cable_array.data.a4[0],
			]
		)

		self.polynomial_heart = Poly(
			[
				cable_array.data.b0[0],
				cable_array.data.b1[0],
				cable_array.data.b2[0],
				cable_array.data.b3[0],
				cable_array.data.b4[0],
			]
		)

	def add_weather_array(self, weather_array: WeatherArray) -> None:
		self.ice_thickness = weather_array.data.ice_thickness.to_numpy()
		self.wind_pressure = weather_array.data.wind_pressure.to_numpy()

	def __getitem__(self, key):
		return getattr(self, key)


def factory_data_container(
	section_array: SectionArray,
	cable_array: CableArray,
	weather_array: WeatherArray,
) -> DataContainer:
	data_container = DataContainer()
	data_container.add_section_array(section_array)
	data_container.add_cable_array(cable_array)
	data_container.add_weather_array(weather_array)
	return data_container
