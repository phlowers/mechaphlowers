# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd

from mechaphlowers.data.initializer.array_importer import (
    ImporterRte,
    import_data_from_proto,
)
from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
    WeatherArray,
)


def test_build_importer():
    importer = ImporterRte("section_import_from_proto_utf8.csv")
    imported_section_array = importer.section_array
    imported_cable_array = importer.cable_array
    imported_weather_array = importer.weather_array
    expected_weather_array = WeatherArray(
        pd.DataFrame(
            {
                "ice_thickness": [1] * 19,
                "wind_pressure": [200] * 19,
            }
        )
    )

    expected_span_lengths = np.array(
        [
            473.07,
            424.53,
            453.46,
            500.07,
            508.96,
            496.89,
            522.35,
            426.38,
            367.72,
            452.33,
            1083.08,
            468.62,
            411.67,
            574.32,
            439.58,
            550,
            329.26,
            544.04,
            516.94,
            np.nan,
        ]
    )

    assert importer.nb_spans == 19
    np.testing.assert_allclose(
        imported_section_array.to_numpy()["span_length"], expected_span_lengths
    )

    assert importer.name_cable == "ASTER600"
    assert imported_cable_array.to_numpy()["section"] == 600.4e-6
    pd.testing.assert_frame_equal(
        imported_weather_array.data, expected_weather_array.data
    )
    np.testing.assert_allclose(importer.new_temperature, 15)


def test_import_data_from_proto():
    section_array, cable_array, weather_array = import_data_from_proto(
        "section_import_from_proto_utf8.csv"
    )
    assert isinstance(section_array, SectionArray)
    assert isinstance(cable_array, CableArray)
    assert isinstance(weather_array, WeatherArray)
