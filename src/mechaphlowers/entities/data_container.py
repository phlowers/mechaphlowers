import numpy as np
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
    WeatherArray,
)


class DataContainer:
    """This class contains data from SectionArray, CableArray and WeatherArray.
    It allows SectionDataFrame to store all data in one class instead of three separate classes.
    Data is stored as attributes, allowing the use of .__dict__() method.

    """

    def __init__(self) -> None:
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

    def update_from_dict(self, data: dict) -> None:
        """Update the attributes of the instance based on a dictionary.

        Args:
                data (dict): Dictionary containing attribute names as keys and their values.
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def add_section_array(self, section_array: SectionArray) -> None:
        """Take as argument a SectionArray, and add all data into its attributes"""
        self.support_name = section_array.data.name.to_numpy()
        self.suspension = section_array.data.suspension.to_numpy()
        self.conductor_attachment_altitude = (
            section_array.data.conductor_attachment_altitude.to_numpy()
        )
        self.crossarm_length = section_array.data.crossarm_length.to_numpy()
        self.line_angle = section_array.data.line_angle.to_numpy()
        self.insulator_length = section_array.data.insulator_length.to_numpy()
        self.insulator_weight = section_array.data.insulator_weight.to_numpy()
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
        """Take as argument a CableArray, and add all data into its attributes.
        CableArray having only one row, we only keep data as np.float.
        The stress-strain polynomial is created, the coefficients are not kept.

        Args:
                cable_array (CableArray): the CableArray that contains data.

        Raises:
                NotImplementedError: raises error if CableArray does not have exactly one row.
        """

        if len(cable_array.data.section) != 1:
            raise NotImplementedError("CableArray should only contain one row")
        self.cable_section_area = cable_array.data.section.iloc[0]
        self.diameter = cable_array.data.diameter.iloc[0]
        self.linear_weight = cable_array.data.linear_weight.iloc[0]
        self.young_modulus = cable_array.data.young_modulus.iloc[0]
        self.dilatation_coefficient = (
            cable_array.data.dilatation_coefficient.iloc[0]
        )
        self.temperature_reference = (
            cable_array.data.temperature_reference.iloc[0]
        )

        self.polynomial_conductor = Poly(
            [
                cable_array.data.a0.iloc[0],
                cable_array.data.a1.iloc[0],
                cable_array.data.a2.iloc[0],
                cable_array.data.a3.iloc[0],
                cable_array.data.a4.iloc[0],
            ]
        )

        self.polynomial_heart = Poly(
            [
                cable_array.data.b0.iloc[0],
                cable_array.data.b1.iloc[0],
                cable_array.data.b2.iloc[0],
                cable_array.data.b3.iloc[0],
                cable_array.data.b4.iloc[0],
            ]
        )

    def add_weather_array(self, weather_array: WeatherArray) -> None:
        """Take as argument a WeatherArray, and add all data into its attributes"""
        self.ice_thickness = weather_array.data.ice_thickness.to_numpy()
        self.wind_pressure = weather_array.data.wind_pressure.to_numpy()


def factory_data_container(
    section_array: SectionArray,
    cable_array: CableArray,
    weather_array: WeatherArray,
) -> DataContainer:
    """Function that creates a DataContainer from arrays.

    Args:
            section_array (SectionArray): SectionArray
            cable_array (CableArray): CableArray
            weather_array (WeatherArray): WeatherArray

    Returns:
            DataContainer: DataContainer instance that contains data from the input arrays
    """
    data_container = DataContainer()
    data_container.add_section_array(section_array)
    data_container.add_cable_array(cable_array)
    data_container.add_weather_array(weather_array)
    return data_container
