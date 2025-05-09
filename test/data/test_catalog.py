# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mechaphlowers.data.catalog import fake_catalog, sample_cable_catalog


def test_fake_catalog__get_mistyping() -> None:
    with pytest.raises(KeyError):
        fake_catalog.get(["wrong_key"])


def test_fake_catalog__get_one_row() -> None:
    first_row = fake_catalog.get(["Bulbasaur"])
    expected_dataframe = pd.DataFrame(
        {
            "#": [1],
            "Type 1": ["Grass"],
            "Type 2": ["Poison"],
            "Total": [318],
            "HP": [45],
            "Attack": [49],
            "Defense": [49],
            "Sp. Atk": [65],
            "Sp. Def": [65],
            "Speed": [45],
            "Generation": [1],
            "Legendary": [False],
        },
        index=["Bulbasaur"],
    )
    expected_dataframe.index.name = "Name"
    assert_frame_equal(first_row, expected_dataframe)


def test_fake_catalog__get_several_rows() -> None:
    # Get first row, row from the middle and last row
    rows = fake_catalog.get(["Bulbasaur", "Spheal", "Volcanion"])
    expected_dataframe = pd.DataFrame(
        {
            "#": [1, 363, 721],
            "Type 1": ["Grass", "Ice", "Fire"],
            "Type 2": ["Poison", "Water", "Water"],
            "Total": [318, 290, 600],
            "HP": [45, 70, 80],
            "Attack": [49, 40, 110],
            "Defense": [49, 50, 120],
            "Sp. Atk": [65, 55, 130],
            "Sp. Def": [65, 50, 90],
            "Speed": [45, 25, 70],
            "Generation": [1, 3, 6],
            "Legendary": [False, False, True],
        },
        index=["Bulbasaur", "Spheal", "Volcanion"],
    )
    expected_dataframe.index.name = "Name"
    assert_frame_equal(rows, expected_dataframe)


def test_fake_catalog__get_same_row_twice() -> None:
    rows = fake_catalog.get(["Spheal", "Spheal"])
    expected_dataframe = pd.DataFrame(
        {
            "#": [363, 363],
            "Type 1": ["Ice", "Ice"],
            "Type 2": ["Water", "Water"],
            "Total": [290, 290],
            "HP": [70, 70],
            "Attack": [40, 40],
            "Defense": [50, 50],
            "Sp. Atk": [55, 55],
            "Sp. Def": [50, 50],
            "Speed": [25, 25],
            "Generation": [3, 3],
            "Legendary": [False, False],
        },
        index=["Spheal", "Spheal"],
    )
    expected_dataframe.index.name = "Name"
    assert_frame_equal(rows, expected_dataframe)


def test_fake_catalog__get_nothing() -> None:
    df = fake_catalog.get([])
    assert df.empty


def test_sample_cable_catalog__get_as_cable_array() -> None:
    cable_array = sample_cable_catalog.get_as_cable_array(
        ["ASTER600", "PETUNIA600"]
    )

    # columns not defined in CableArrayInput should be dropped
    assert "data_source" not in cable_array.data

    cable_array.data.section
    cable_array.data.index


def test_sample_cable_catalog__get_as_cable_array__missing_key() -> None:
    with pytest.raises(KeyError):
        sample_cable_catalog.get_as_cable_array(["wrong_key"])


def test_fake_catalog__get_one_row_ot_list() -> None:
    # string case
    first_row = fake_catalog.get("Bulbasaur")
    expected_dataframe = pd.DataFrame(
        {
            "#": [1],
            "Type 1": ["Grass"],
            "Type 2": ["Poison"],
            "Total": [318],
            "HP": [45],
            "Attack": [49],
            "Defense": [49],
            "Sp. Atk": [65],
            "Sp. Def": [65],
            "Speed": [45],
            "Generation": [1],
            "Legendary": [False],
        },
        index=["Bulbasaur"],
    )
    expected_dataframe.index.name = "Name"
    assert_frame_equal(first_row, expected_dataframe)
    # other cases
    with pytest.raises(TypeError):
        first_row = fake_catalog.get(123)  # type: ignore[arg-type]


def test_fake_catalog__keys() -> None:
    """Test the `keys` method of the FakeCatalog class"""
    assert len(fake_catalog.keys()) == 800
    assert "Bulbasaur" in fake_catalog.keys()
    assert "notPokemon" not in fake_catalog.keys()
