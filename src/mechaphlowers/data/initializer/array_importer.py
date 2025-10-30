# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from mechaphlowers.data.catalog.catalog import sample_cable_catalog
from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
    WeatherArray,
)

DATA_BASE_PATH = Path(__file__).absolute().parent


class Importer(ABC):
    """Base class for importers."""

    @property
    @abstractmethod
    def section_array(self) -> SectionArray:
        """Get SectionArray from csv file"""

    @property
    @abstractmethod
    def cable_array(self) -> CableArray:
        """Get CableArray from csv file"""

    @property
    @abstractmethod
    def weather_array(self) -> WeatherArray:
        """Get Weather from csv file"""


class ImporterRte(Importer):
    """Importer for RTE data."""

    translation_map_fr = {
        "nom": "name",
        "suspension": "suspension",
        "alt_acc": "conductor_attachment_altitude",
        "long_bras": "crossarm_length",
        "angle_ligne": "line_angle",
        "long_ch": "insulator_length",
        "portée": "span_length",
        "pds_ch": "insulator_mass",
    }

    def __init__(self, filename: str | PathLike) -> None:
        filepath = DATA_BASE_PATH / filename

        self.raw_df = pd.read_csv(
            filepath,
            decimal=",",
            sep=";",
            encoding="utf-8",
            dtype={"nom": str, "portée": float},
            index_col=0,
        )

        self.get_data_last_column()

    # new_temperature is not extracted but can be accessed through ImporterRte
    def get_data_last_column(self) -> None:
        last_column = list(self.raw_df["nb_portées"])
        self.data_numeric_values = {
            "sagging_temperature": float(last_column[6]),
            "sagging_parameter": float(last_column[8]),
            "new_temperature": float(last_column[12]),
            "wind_pressure": float(last_column[14]),
            "ice_thickness": float(last_column[16]),
        }
        self.nb_spans = int(last_column[0])
        self.name_cable = str(last_column[2])

    @property
    def section_array(self) -> SectionArray:
        renamed_df = self.raw_df.rename(columns=self.translation_map_fr)
        # Convert VRAI/FAUX from french .xlsx file into pythonic True/False
        renamed_df["suspension"] = renamed_df["suspension"].map(
            lambda value: True if value == "VRAI" else False
        )

        # change sign of crossarm_length and line_angle to match mechaphlowers (anticlockwise sense)
        renamed_df["crossarm_length"] = -renamed_df["crossarm_length"]
        renamed_df["line_angle"] = -renamed_df["line_angle"]

        section_array = SectionArray(renamed_df)
        # convert line_angle from grad to degrees
        section_array.add_units({"line_angle": "grad"})
        section_array.sagging_parameter = self.data_numeric_values[
            "sagging_parameter"
        ]
        section_array.sagging_temperature = self.data_numeric_values[
            "sagging_temperature"
        ]
        return section_array

    @property
    def cable_array(self) -> CableArray:
        return sample_cable_catalog.get_as_object([self.name_cable])  # type: ignore

    @property
    def weather_array(self) -> WeatherArray:
        ice_thickness = self.data_numeric_values["ice_thickness"]
        wind_pressure = self.data_numeric_values["wind_pressure"]
        weather_array = WeatherArray(
            pd.DataFrame(
                {
                    "ice_thickness": [ice_thickness] * self.nb_spans,
                    "wind_pressure": [wind_pressure] * self.nb_spans,
                }
            )
        )
        return weather_array

    @property
    def new_temperature(self) -> np.ndarray:
        new_temperature_float = self.data_numeric_values["new_temperature"]
        return np.array([new_temperature_float] * self.nb_spans)


def import_data_from_proto(
    filename: str | PathLike,
) -> Tuple[SectionArray, CableArray, WeatherArray]:
    importer = ImporterRte(filename)
    return importer.section_array, importer.cable_array, importer.weather_array
