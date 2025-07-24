# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
import warnings
from os import PathLike
from pathlib import Path
from typing import Literal, get_args

import pandas as pd
import pandera as pa
import yaml  # type: ignore[import-untyped]

from mechaphlowers.entities.arrays import (
    CableArray,
    ElementArray,
    WeatherArray,
)

# Resolve the 'data' folder
# which is the parent folder of this script
# in order to later be able to find the data files stored in this folder.
DATA_BASE_PATH = Path(__file__).absolute().parent

logger = logging.getLogger(__name__)

CatalogType = Literal['', "cable_catalog", "weather_catalog"]
list_object_conversion = [None, CableArray, WeatherArray]
catalog_to_object = dict(
    zip(list(get_args(CatalogType)), list_object_conversion)
)


class Catalog:
    """Generic wrapper for tabular data read from a csv file, indexed by a `key` column."""

    def __init__(
        self,
        filename: str | PathLike,
        key_column_name: str,
        catalog_type: CatalogType = '',
        columns_types: dict | None = None,
        rename_dict: dict | None = None,
    ) -> None:
        """Initialize catalog from a csv file.

        For now, we only support csv input files located in the `data/` folder of the source code.

        Please note that the responsibility of ensuring the "uniqueness" of the `key` column is left
        to the user: no integrity check is performed on input data.

        Args:
                filename (str | PathLike): filename of the csv data source
                key_column_name (str): name of the column used as key (i.e. row identifier)
        """
        self.catalog_type = catalog_type
        if columns_types is None:
            columns_types = {}
        if rename_dict is None:
            rename_dict = {}
        filepath = DATA_BASE_PATH / filename
        df_schema = pa.DataFrameSchema(
            {key: pa.Column(value) for (key, value) in columns_types.items()},
        )
        # forcing key index to be a str. Key index should not be in types_dict
        columns_types[key_column_name] = 'str'
        self._data = pd.read_csv(
            filepath, index_col=key_column_name, dtype=columns_types
        )

        # validating the pandera schema. Useful for checking missing fields
        df_schema.validate(self._data)
        self.rename_columns(key_column_name, rename_dict)
        self.remove_duplicates(filename)

    def rename_columns(self, key_column_name, rename_dict):
        self._data = self._data.rename(columns=rename_dict)
        # also renaming index column
        if key_column_name in rename_dict:
            self._data.index.names = [rename_dict[key_column_name]]

    def remove_duplicates(self, filename):
        # removing duplicates, and warn if any duplicates found
        duplicated = self._data.index.duplicated()
        if duplicated.any():
            self._data = self._data[~duplicated]
            logger.warning(f'Duplicate key index found for catalog {filename}')
            warnings.warn(
                f'Duplicate key index found for catalog {filename}',
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

    def get_as_object(self, keys: list) -> ElementArray:
        """Get rows from a list of keys.

        If a key is present several times in the `keys` argument, the returned dataframe
        will contain the corresponding row as many times as requested.

        If any of the requested `keys` were to match several rows, all matching rows would
        be returned.

        The type of the object returned depends on catalog_type.
        The mapping between catalog_type and object type is made by dictionnary catalog_to_object

        Raises:
                KeyError: if any of the requested `keys` doesn't match any row in the input data

        Args:
                keys (list): list of keys

        Returns:
                object: requested object, that depends on `catalog_type`
        """
        try:
            catalog_to_object[self.catalog_type]
        except KeyError:
            raise KeyError(
                f"Catalog type '{self.catalog_type}' is not supported. "
                "Supported types are: "
                f"{list(catalog_to_object.keys())}"
            )
        df = self.get(keys)
        return catalog_to_object[self.catalog_type](df)

    def keys(self) -> list:
        """Get the keys available in the catalog"""
        return self._data.index.tolist()

    def __str__(self) -> str:
        return self._data.to_string()


def build_catalog_from_yaml(
    yaml_filename: str | PathLike, rename=True
) -> Catalog:
    """Build a catalog from a yaml file.

    Args:
        yaml_filename (str | PathLike): path to the yaml file

    Returns:
        Catalog: a catalog instance with the data from the yaml file
    """

    yaml_filepath = DATA_BASE_PATH / yaml_filename
    with open(yaml_filepath, "r") as file:
        data = yaml.safe_load(file)

    # fetch data for type validation
    columns_types = {
        key: value
        for list_item in data["columns"]
        for (key, value) in list_item.items()
    }

    # fetch data for renaming columns
    if rename:
        rename_dict = {
            key: value
            for list_item in data["columns_mapping"]
            for (key, value) in list_item.items()
        }
    else:
        rename_dict = {}
    catalog_type = data["catalog_type"]
    return Catalog(
        data["csv_path"],
        data["key_column_name"],
        catalog_type,
        columns_types,
        rename_dict,
    )


fake_catalog = Catalog("pokemon.csv", key_column_name="Name")
sample_cable_catalog = build_catalog_from_yaml("sample_cable_database.yaml")
