# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
import pytest
import yaml
from pandas.testing import assert_frame_equal

from mechaphlowers.data.catalog.catalog import (
    Catalog,
    build_catalog_from_yaml,
    fake_catalog,
    sample_cable_catalog,
    sample_support_catalog,
    write_yaml_catalog_template,
)
from mechaphlowers.entities.arrays import CableArray, ElementArray
from mechaphlowers.entities.shapes import SupportShape


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
    cable_array: ElementArray = sample_cable_catalog.get_as_object(
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


def test_support_catalog():
    aaa = sample_support_catalog.get_as_object("support_name_1")
    bbb = sample_support_catalog.get_as_object(
        ["support_name_1", "support_name_5"]
    )

    assert len(bbb) == 2
    assert len(aaa) == 1
    assert aaa[0].name == "support_name_1"

    np.testing.assert_allclose(
        aaa[0].trunk_points, np.array([[0, 0, 0], [0, 0, 23.0]])
    )

    assert len(bbb[1].labels_points) == 7
    assert len(bbb[0].labels_points) == 4

    assert len(bbb[1].support_points) == (7 + 1) * 3

    with pytest.raises(IndexError):
        SupportShape.from_dataframe(pd.DataFrame([]))


def test_custom_catalog_loading(tmp_path):
    CONTENT_YML = {
        'csv_name': 'pokemon.csv',
        'catalog_type': 'default_catalog',
        'key_column_name': 'Name',
        'columns': [
            {'Type 1': 'str'},
            {'Type 2': 'str'},
            {'Total': 'int'},
            {'HP': 'int'},
            {'Attack': 'int'},
            {'Defense': 'int'},
            {'Sp. Atk': 'int'},
            {'Sp. Def': 'int'},
            {'Speed': 'int'},
            {'Generation': 'int'},
            {'Legendary': 'bool'},
        ],
        'columns_renaming': [
            {'Type 1': 'First Type'},
            {'Type 2': 'Second Type'},
        ],
    }

    CONTENT_CSV = """#,Name,Type 1,Type 2,Total,HP,Attack,Defense,Sp. Atk,Sp. Def,Speed,Generation,Legendary
    1,Bulbasaur,Grass,Poison,318,45,49,49,65,65,45,1,False
    2,Ivysaur,Grass,Poison,405,60,62,63,80,80,60,1,False"""

    d = tmp_path / "user_folder"
    d.mkdir()

    with pytest.raises(FileNotFoundError):
        build_catalog_from_yaml("not_a_file.yaml")

    file_path_yaml = d / "custom_catalog.yaml"

    with open(file_path_yaml, 'w') as file:
        yaml.dump(CONTENT_YML, file)

    with pytest.raises(FileNotFoundError):
        build_catalog_from_yaml("custom_catalog.yaml", user_filepath=d)

    file_path_csv = d / "pokemon.csv"
    file_path_csv.write_text(CONTENT_CSV, encoding="utf-8")
    catalog = build_catalog_from_yaml("custom_catalog.yaml", user_filepath=d)
    c_line = catalog.get(["Bulbasaur", "Ivysaur"])
    assert isinstance(c_line, pd.DataFrame)
    assert catalog._data.shape == (2, 12)


def test_write_yaml_catalog_template_support(tmp_path):
    write_yaml_catalog_template(tmp_path, template="support_catalog")
    expected_file = tmp_path / "sample_pylon_database.yaml"
    assert expected_file.exists()
    with open(expected_file) as f:
        content = f.read()
        assert "support_catalog" in content


def test_write_yaml_catalog_template_cable(tmp_path):
    write_yaml_catalog_template(tmp_path, template="cable_catalog")
    expected_file = tmp_path / "sample_cable_database.yaml"
    assert expected_file.exists()
    with open(expected_file) as f:
        content = f.read()
        assert "cable_catalog" in content


def test_write_yaml_catalog_template_invalid_template(tmp_path):
    with pytest.raises(KeyError):
        write_yaml_catalog_template(tmp_path, template="unknown_catalog")


def test_write_yaml_catalog_template_invalid_path():
    # Use a path that does not exist
    invalid_path = Path("/unlikely/to/exist/for/test")
    with pytest.raises(FileNotFoundError):
        write_yaml_catalog_template(invalid_path, template="support_catalog")
    with pytest.raises(TypeError):
        write_yaml_catalog_template(1, template="cable_catalog")


def test_write_yaml_catalog_template_str_path(tmp_path):
    # Accepts str as path
    write_yaml_catalog_template(str(tmp_path), template="support_catalog")
    expected_file = tmp_path / "sample_pylon_database.yaml"
    assert expected_file.exists()


def test_catalog_cable_array_units_df():
    cable_array = sample_cable_catalog.get_as_object(["ASTER600"])

    expected_result_SI_units = pd.DataFrame(
        {
            "section": [600.4e-6],
            "diameter": [31.86e-3],
            # TODO: differeciate linear_mass/linear_weight
            "linear_weight": [17.658],
            "young_modulus": [60e9],
            "dilatation_coefficient": [23e-6],
            "temperature_reference": [15.0],
            "a0": [0.0],
            "a1": [60e9],
            "a2": [0.0],
            "a3": [0.0],
            "a4": [0.0],
            "b0": [0.0],
            "b1": [0.0],
            "b2": [0.0],
            "b3": [0.0],
            "b4": [0.0],
        }
    )
    # check dtype?
    assert_frame_equal(
        cable_array.data.reset_index(drop=True),
        expected_result_SI_units,
        check_like=True,
        atol=1e-07,
    )


def test_catalog_cable_array_units_object():
    cable_array = sample_cable_catalog.get_as_object(["ASTER600"])

    cable_array_original = CableArray(
        pd.DataFrame(
            {
                "section": [600.4],
                "diameter": [31.86],
                "linear_mass": [1.8],
                "young_modulus": [60],
                "dilatation_coefficient": [23],
                "temperature_reference": [15],
                "a0": [0],
                "a1": [60],
                "a2": [0],
                "a3": [0],
                "a4": [0],
                "b0": [0],
                "b1": [0],
                "b2": [0],
                "b3": [0],
                "b4": [0],
            }
        )
    )

    # check dtype?
    assert_frame_equal(
        cable_array.data.reset_index(drop=True),
        cable_array_original.data.reset_index(drop=True),
        check_like=True,
        atol=1e-07,
    )
