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
		self.b0: np.ndarray
		self.b1: np.ndarray
		self.b2: np.ndarray
		self.b3: np.ndarray
		self.b4: np.ndarray

		self.polynomial_conductor: np.ndarray
		self.polynomial_heart: np.ndarray

		self.ice_thickness: np.ndarray
		self.wind_pressure: np.ndarray

	@classmethod
	def init_with_dict(cls, input_dict: dict):
		data_container = cls()
		data_container.__dict__ = input_dict
		return data_container

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
		b0: np.ndarray,
		b1: np.ndarray,
		b2: np.ndarray,
		b3: np.ndarray,
		b4: np.ndarray,
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

		poly_conductor_array = [
			Poly([a0[index], a1[index], a2[index], a3[index], a4[index]])
			for index in range(len(a0))
		]
		self.polynomial_conductor = np.array(poly_conductor_array)

		poly_heart_array = [
			Poly([b0[index], b1[index], b2[index], b3[index], b4[index]])
			for index in range(len(b0))
		]
		self.polynomial_heart = np.array(poly_heart_array)

	def add_data_weather(
		self, ice_thickness: np.ndarray, wind_pressure: np.ndarray, **kwargs
	) -> None:
		self.ice_thickness = ice_thickness
		self.wind_pressure = wind_pressure

	def add_section_array(self, section_array: SectionArray) -> None:
		self.name = section_array.data.name.to_numpy()
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
		self.section = cable_array.data.section.to_numpy()
		self.diameter = cable_array.data.diameter.to_numpy()
		self.linear_weight = cable_array.data.linear_weight.to_numpy()
		self.young_modulus = cable_array.data.young_modulus.to_numpy()
		self.dilatation_coefficient = (
			cable_array.data.dilatation_coefficient.to_numpy()
		)
		self.temperature_reference = (
			cable_array.data.temperature_reference.to_numpy()
		)
		self.a0 = cable_array.data.a0.to_numpy()
		self.a1 = cable_array.data.a1.to_numpy()
		self.a2 = cable_array.data.a2.to_numpy()
		self.a3 = cable_array.data.a3.to_numpy()
		self.a4 = cable_array.data.a4.to_numpy()
		self.b0 = cable_array.data.b0.to_numpy()
		self.b1 = cable_array.data.b1.to_numpy()
		self.b2 = cable_array.data.b2.to_numpy()
		self.b3 = cable_array.data.b3.to_numpy()
		self.b4 = cable_array.data.b4.to_numpy()

		poly_conductor_array = [
			Poly(
				[
					self.a0[index],
					self.a1[index],
					self.a2[index],
					self.a3[index],
					self.a4[index],
				]
			)
			for index in range(len(self.a0))
		]
		self.polynomial_conductor = np.array(poly_conductor_array)

		poly_heart_array = [
			Poly(
				[
					self.b0[index],
					self.b1[index],
					self.b2[index],
					self.b3[index],
					self.b4[index],
				]
			)
			for index in range(len(self.b0))
		]
		self.polynomial_heart = np.array(poly_heart_array)

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
	data_container.add_data_section(**section_array.to_dict_with_numpy())
	data_container.add_data_cable(**cable_array.to_dict_with_numpy())
	data_container.add_data_weather(**weather_array.to_dict_with_numpy())
	return data_container
