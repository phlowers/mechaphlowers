# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import warnings

import numpy as np
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
    cable_array = sample_cable_catalog.get_as_object(
        ["ASTER600", "PETUNIA600"]
    )

    # columns not defined in CableArrayInput should be dropped
    assert "data_source" not in cable_array.data

    cable_array.data.section
    cable_array.data.index


def test_fake_catalog__get_as_object() -> None:
    with pytest.raises(KeyError):
        fake_catalog.get_as_object(["Bulbasaur"])


def test_sample_cable_catalog__get_as_cable_array__missing_key() -> None:
    with pytest.raises(KeyError):
        sample_cable_catalog.get_as_object(["wrong_key"])


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
    assert True


def test_fake_catalog_rename():
    rename_dict = {
        "Name": "Nom",
        "Attack": "Attaque",
        "Speed": "Vitesse",
        "Generation": "Génération",
        "Legendary": "Légendaire",
    }
    pkmn_catalog = Catalog(
        "pokemon.csv", key_column_name="Name", rename_dict=rename_dict
    )
    translated_columns = {"Attaque", "Vitesse", "Génération", "Légendaire"}

    assert translated_columns.issubset(set(pkmn_catalog._data.columns))
    assert pkmn_catalog._data.index.names == ["Nom"]


def test_type_valdiation():
    types_dict = {
        "Attack": int,
        "Speed": float,
        "Generation": int,
        "Legendary": bool,
    }
    Catalog("pokemon.csv", key_column_name="Name", columns_types=types_dict)


def test_fake_catalog_type_checking__missing_arg():
    types_dict = {"wrong_arg": int}
    with pytest.raises(pa.errors.SchemaError):
        Catalog(
            "pokemon.csv", key_column_name="Name", columns_types=types_dict
        )


# This test should check be decommentated when fixing the fact that bool are not validated
# def test__read_csv__wrong_type():
#     types_dict = {
#         "Speed": bool,
#     }
#     with pytest.raises(ValueError):
#         Catalog(
#             "pokemon.csv", key_column_name="Name", columns_types=types_dict
#         )


@pytest.mark.filterwarnings("ignore::UserWarning")
def test__read_csv__manage_empty_bool():
    types_dict = {
        "boolean arg": bool,
    }
    catalog = Catalog(
        "iris_dataset.csv",
        key_column_name="sepal length (cm)",
        columns_types=types_dict,
    )
    assert np.isnan(catalog._data.loc["5.1", "boolean arg"])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_iris_catalog__drop_duplicates():
    iris_catalog = Catalog(
        "iris_dataset.csv",
        key_column_name="sepal length (cm)",
        columns_types={},
    )
    extract_df = iris_catalog.get("5.1")
    assert len(extract_df) == 1


def test_duplicated_warning():
    with warnings.catch_warnings(record=True) as warning:
        Catalog("iris_dataset.csv", key_column_name="sepal length (cm)")
        assert len(warning) == 1
        assert warning[0].category is UserWarning
        assert "iris_dataset.csv" in str(warning[0].message)
