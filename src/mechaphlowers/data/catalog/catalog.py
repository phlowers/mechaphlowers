# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
import warnings
from os import PathLike
from pathlib import Path
from typing import Any, Literal, get_args

import pandas as pd
import pandera as pa
import yaml  # type: ignore[import-untyped]

from mechaphlowers.entities.arrays import (
    CableArray,
    ElementArray,
)
from mechaphlowers.entities.shapes import SupportShape

# Resolve the 'data' folder
# which is the parent folder of this script
# in order to later be able to find the data files stored in this folder.
DATA_BASE_PATH = Path(__file__).absolute().parent

logger = logging.getLogger(__name__)


CatalogType = Literal['default_catalog', 'cable_catalog', 'support_catalog']
list_object_conversion = [None, CableArray, SupportShape.from_dataframe]
catalog_to_object = dict(
    zip(list(get_args(CatalogType)), list_object_conversion)
)


class Catalog:
    """Generic wrapper for tabular data read from a csv file, indexed by a `key` column."""

    def __init__(
        self,
        filename: str | PathLike,
        key_column_name: str,
        columns_types: dict,
        rename_dict: dict,
        units_dict: dict,
        catalog_type: CatalogType = 'default_catalog',
        remove_duplicates: bool = True,
        user_filepath: str | PathLike = DATA_BASE_PATH,
    ) -> None:
        """Initialize catalog from a csv file.

        For now, we only support csv input files located in the `data/` folder of the source code.

        Please note that the responsibility of ensuring the "uniqueness" of the `key` column is left
        to the user: no integrity check is performed on input data.

        Args:
                filename (str | PathLike): filename of the csv data source
                key_column_name (str): name of the column used as key (i.e. row identifier)
                catalog_type (Literal['default_catalog', "cable_catalog"]): type of the catalog. Used in the `get_as_object` method to convert the catalog to a specific object type.
                columns_types (dict | None): dictionary of column names and their types.
                rename_dict (dict | None): dictionary of column names to rename. The key is the original name, the value is the new name.
                remove_duplicates (bool): whether to remove duplicate rows. Defaults to True.
                user_filepath (str | PathLike): path to the folder containing the csv file. Defaults to internal data.
        """
        self.catalog_type = catalog_type
        self.units_dict = units_dict
        if user_filepath is None:
            filepath = DATA_BASE_PATH / filename  # type: ignore[operator]
        else:
            filepath = Path(user_filepath) / filename  # type: ignore[operator]
        if not filepath.exists():
            raise FileNotFoundError(
                f"{user_filepath=} seems to be not a valid path. Please provide a valid path."
            )

        # Warning: booleans are not treated correctly in order to avoid issues with empty values.
        # TODO: Maybe remove this filter if we consider that empty values on boolean columns does not exist, or fix this
        dtype_dict_without_bool = {
            key: value
            for (key, value) in columns_types.items()
            if value is not bool
        }

        # forcing key index to be a str. Key index should not be in types_dict
        dtype_dict_with_key = dtype_dict_without_bool.copy()
        dtype_dict_with_key[key_column_name] = str

        self._data = pd.read_csv(
            filepath,
            index_col=key_column_name,
            dtype=dtype_dict_with_key,
        )
        # validating the pandera schema. Useful for checking missing fields
        self.validate_types(dtype_dict_without_bool)
        self.rename_columns(key_column_name, rename_dict)
        if remove_duplicates is True:
            self.remove_duplicates(filename)

    def validate_types(self, dtype_dict: dict) -> None:
        """Validate the types of the dataframe. Boolean columns are not checked.

        Args:
            dtype_dict (dict): dictionary of column names and their types, without the key index and the boolean columns.
        """
        coerce_dict = {
            str: True,
            int: True,
            float: True,
            bool: False,
        }
        df_schema = pa.DataFrameSchema(
            {
                key: pa.Column(value, nullable=True, coerce=coerce_dict[value])
                for (key, value) in dtype_dict.items()
            },
            index=pa.Index(str),
        )
        df_schema.validate(self._data)

    def rename_columns(self, key_column_name: str, rename_dict: dict) -> None:
        """Rename the columns and the index of the catalog

        Args:
            key_column_name (str): name of the key index
            rename_dict (dict): dictionary of all column names that need to be renamed. This can include the key index.
        """
        self._data = self._data.rename(columns=rename_dict)
        # also renaming index column
        if key_column_name in rename_dict:
            self._data.index.names = [rename_dict[key_column_name]]

    def remove_duplicates(self, filename: str | PathLike) -> None:
        """Remove duplicate rows, and warn if any duplicates found.

        Args:
            filename (str | PathLike): filename of the csv data source (used only for logging a warning)
        """
        # removing duplicate rows, and warn if any duplicates found
        duplicated = self._data.index.duplicated()
        if duplicated.any():
            self._data = self._data[~duplicated]
            logger.warning(
                f'Duplicate key indices found for catalog {filename}'
            )
            warnings.warn(
                f'Duplicate key indices found for catalog {filename}',
                UserWarning,
            )

    def get(self, keys: list | str) -> pd.DataFrame:
        """Get rows from a list of keys.

        If a key is present several times in the `keys` argument, the returned dataframe
        will contain the corresponding row as many times as requested.

        If any of the requested `keys` were to match several rows, all matching rows would
        be returned.

        Raises:
                KeyError: if any of the requested `keys` doesn't match any row in the input data

        Args:
                keys (list): list of keys

        Returns:
                pd.DataFrame: requested rows
        """
        if isinstance(keys, str):
            keys = [keys]
        elif not isinstance(keys, list):
            raise TypeError(
                f"Expected a list or str as argument for 'keys', got {type(keys)}"
            )
        try:
            return self._data.loc[keys]
        except KeyError as e:
            raise KeyError(
                f"Error when requesting catalog: {e.args[0]}. Try the .keys() method to gets the available keys?"
            ) from e

    def get_as_object(self, keys: list) -> Any:
        """Get rows from a list of keys.

        If a key is present several times in the `keys` argument, the returned dataframe
        will contain the corresponding row as many times as requested.

        If any of the requested `keys` were to match several rows, all matching rows would
        be returned.

        The type of the object returned depends on catalog_type.
        The mapping between catalog_type and object type is made by dictionary catalog_to_object

        Raises:
                KeyError: if any of the requested `keys` doesn't match any row in the input data

        Args:
                keys (list): list of keys

        Returns:
                object: requested object, that depends on `catalog_type`
        """
        if (
            self.catalog_type == "default_catalog"
            or self.catalog_type not in catalog_to_object
        ):
            raise KeyError(
                f"Catalog type '{self.catalog_type}' is not supported for get_as_object(). "
                "Supported types are: "
                f"{list(catalog_to_object.keys())[1:]}"
            )
        df = self.get(keys)
        element_array = catalog_to_object[self.catalog_type](df)
        if isinstance(element_array, ElementArray):
            element_array.add_units(self.units_dict)
        return element_array

    def keys(self) -> list:
        """Get the keys available in the catalog"""
        return self._data.index.tolist()

    def __str__(self) -> str:
        return self._data.to_string()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} object of {self.catalog_type}\n{self.__str__()}"


def build_catalog_from_yaml(
    yaml_filename: str | PathLike,
    rename=True,
    remove_duplicates=True,
    user_filepath: str | PathLike = DATA_BASE_PATH,
) -> Catalog:
    """Build a catalog from a yaml file.

    Args:
        yaml_filename (str | PathLike): path to the yaml file
        rename (bool): whether to rename columns according to the yaml file. Defaults to True.
        remove_duplicates (bool): whether to remove duplicate rows. Defaults to True.
        user_filepath (str | PathLike): path to the folder containing the yaml file. Defaults to internal data.

    Returns:
        Catalog: a catalog instance with the data from the yaml file
    """

    try:
        yaml_filepath = user_filepath / yaml_filename  # type: ignore[operator]

        with open(yaml_filepath, "r") as file:
            raw_data_yaml: dict = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {yaml_filepath} not found")

    string_to_type_converters = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        str: str,
        int: int,
        float: float,
        bool: bool,
    }
    # fetch data for type validation
    if "columns" in raw_data_yaml:
        columns_types = {
            key: string_to_type_converters[value]
            for list_item in raw_data_yaml["columns"]
            for (key, value) in list_item.items()
        }
    else:
        columns_types = {}
    # fetch data for renaming columns
    # TODO: add test case no rename
    rename_dict = (
        fetch_dict_from_yaml("columns_renaming", raw_data_yaml)
        if rename
        else {}
    )
    # fetch data for input units
    units_dict = fetch_dict_from_yaml("columns_units", raw_data_yaml)
    catalog_type = raw_data_yaml["catalog_type"]
    return Catalog(
        raw_data_yaml["csv_name"],
        raw_data_yaml["key_column_name"],
        columns_types,
        rename_dict,
        units_dict,
        catalog_type,
        remove_duplicates,
        user_filepath,
    )


def fetch_dict_from_yaml(dict_name: str, raw_data_yaml: dict) -> dict:
    if dict_name in raw_data_yaml:
        return {
            key: value
            for list_item in raw_data_yaml[dict_name]
            for (key, value) in list_item.items()
        }
    else:
        return {}


def write_yaml_catalog_template(
    user_filepath: str | PathLike,
    template: Literal["cable_catalog", "support_catalog"] = "support_catalog",
) -> None:
    """Write a yaml catalog template file in the user_filepath folder provided.

    Args:
        user_filepath (str | PathLike): path to the folder where the yaml file will be written. Defaults to internal data.
        template (Literal['cable_catalog', 'support_catalog']): type of the catalog template. Defaults to "support_catalog".
    """
    map_catalog = {
        "cable_catalog": "sample_cable_database.yaml",
        "support_catalog": "sample_pylon_database.yaml",
    }

    try:
        filename = map_catalog[template]
    except KeyError as e:
        raise KeyError(
            f"Template '{template}' is not supported. Supported templates are: {list(map_catalog.keys())}"
        ) from e

    if isinstance(user_filepath, str):
        filepath = Path(user_filepath)
    elif isinstance(user_filepath, Path):
        filepath = user_filepath
    else:
        raise TypeError(
            f"Expected a str or Path as argument for 'user_filepath', got {type(user_filepath)}"
        )

    if not filepath.exists():
        raise FileNotFoundError(
            f"{user_filepath=} seems to be not a valid path. Please provide a valid path."
        )

    filepath = filepath / filename  # type: ignore[operator]

    with open(filepath, 'w') as file:
        yaml.dump(template, file)


fake_catalog = build_catalog_from_yaml("pokemon.yaml", rename=False)
sample_cable_catalog = build_catalog_from_yaml("sample_cable_database.yaml")
sample_support_catalog = build_catalog_from_yaml(
    "sample_pylon_database.yaml", remove_duplicates=False
)
