# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path

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

    def get_data_last_column(self) -> None:
        last_column = list(self.raw_df["nb_portées"])
        self.data_last_column = {
            "nb_spans": int(last_column[0]),
            "sagging_temperature": int(last_column[6]),
            "sagging_parameter": int(last_column[8]),
            "new_temperature": int(last_column[12]),
            "wind_pressure": int(last_column[14]),
            "ice_thickness": int(last_column[16]),
        }
        self.name_cable = str(last_column[2])

    @property
    def section_array(self) -> SectionArray:
        renamed_df = self.raw_df.rename(columns=self.translation_map_fr)
        renamed_df["suspension"] = renamed_df["suspension"].map(
            lambda value: True if value == "VRAI" else False
        )

        # convert arm length + angle_line grad -> deg
        renamed_df["crossarm_length"] = -renamed_df["crossarm_length"]
        renamed_df["line_angle"] = -renamed_df["line_angle"] * 0.9

        section_array = SectionArray(renamed_df)

        section_array.sagging_parameter = self.data_last_column[
            "sagging_parameter"
        ]
        section_array.sagging_temperature = self.data_last_column[
            "sagging_temperature"
        ]
        return section_array

    @property
    def cable_array(self) -> CableArray:
        return sample_cable_catalog.get_as_object([self.name_cable])  # type: ignore

    @property
    def weather_array(self) -> WeatherArray:
        ice_thickness = self.data_last_column["ice_thickness"]
        wind_pressure = self.data_last_column["wind_pressure"]
        nb_spans = self.data_last_column["nb_spans"]
        weather_array = WeatherArray(
            pd.DataFrame(
                {
                    "ice_thickness": [ice_thickness] * nb_spans,
                    "wind_pressure": [wind_pressure] * nb_spans,
                }
            )
        )
        return weather_array
