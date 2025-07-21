# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import pandera as pa
import pytest
from pandas.testing import assert_frame_equal

from mechaphlowers.data.catalog import (
    Catalog,
    build_catalog_from_yaml,
    fake_catalog,
    sample_cable_catalog,
)


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


def test_yaml():
    build_catalog_from_yaml("sample_cable_database.yaml")


def test_fake_catalog_rename():
    rename_dict = {
        "Name": "Nom",
        "Attack": "Attaque",
        "Speed": "Vitesse",
        "Generation": "Génération",
        "Legendary": "Légendaire",
    }
    pkmn_catalog = Catalog(
        "pokemon.csv", key_column_name="Name", rename_map=rename_dict
    )
    translated_columns = {"Attaque", "Vitesse", "Génération", "Légendaire"}

    assert translated_columns.issubset(set(pkmn_catalog._data.columns))
    assert pkmn_catalog._data.index.names == ["Nom"]


def test_fake_catalog_type_checking():
    df_schema = pa.DataFrameSchema(
        {
            "Attack": pa.Column(int),
            "Speed": pa.Column(int),
            "Generation": pa.Column(int),
            "Legendary": pa.Column(bool),
        },
        # coerce=True
    )
    Catalog("pokemon.csv", key_column_name="Name", df_schema=df_schema)


def test_fake_catalog_type_checking__wrong_type():
    df_schema = pa.DataFrameSchema(
        {
            "Speed": pa.Column(bool),
        },
    )
    with pytest.raises(pa.errors.SchemaError):
        Catalog("pokemon.csv", key_column_name="Name", df_schema=df_schema)


def test_fake_catalog_type_checking__missing_arg():
    df_schema = pa.DataFrameSchema(
        {
            "wrong_arg": pa.Column(int),
        },
    )
    with pytest.raises(pa.errors.SchemaError):
        Catalog("pokemon.csv", key_column_name="Name", df_schema=df_schema)
