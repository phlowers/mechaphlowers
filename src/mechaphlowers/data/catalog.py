# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from os import PathLike
from pathlib import Path

import pandas as pd
import yaml  # type: ignore[import-untyped]

from mechaphlowers.entities.arrays import CableArray

# Resolve the 'data' folder
# which is the parent folder of this script
# in order to later be able to find the data files stored in this folder.
DATA_BASE_PATH = Path(__file__).absolute().parent


class Catalog:
    """Generic wrapper for tabular data read from a csv file, indexed by a `key` column."""

    def __init__(
        self,
        filename: str | PathLike,
        key_column_name: str,
        rename_map: dict = {},
    ) -> None:
        """Initialize catalog from a csv file.

        For now, we only support csv input files located in the `data/` folder of the source code.

        Please note that the responsibility of ensuring the "uniqueness" of the `key` column is left
        to the user: no integrity check is performed on input data.

        Args:
                filename (str | PathLike): filename of the csv data source
                key_column_name (str): name of the column used as key (i.e. row identifier)
        """
        filepath = DATA_BASE_PATH / filename
        self._data = pd.read_csv(filepath, index_col=key_column_name)
        self._data = self._data.rename(columns=rename_map)

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

    def get_as_cable_array(self, keys: list) -> CableArray:
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
                CableArray: requested rows
        """
        df = self.get(keys)
        return CableArray(df)
        # TODO(ai-qui): make this generic (CableArray vs. generic Catalog)?

    def keys(self) -> list:
        """Get the keys available in the catalog"""
        return self._data.index.tolist()

    def __str__(self) -> str:
        return self._data.to_string()


# keep this?
def build_catalog(filename: str | PathLike, key_column_name: str) -> Catalog:
    """Build a catalog from the default data files.

    Returns:
            Catalog: a catalog instance with the default data files
    """
    return Catalog(filename, key_column_name)


def build_catalog_from_yaml(
    yaml_filename: str | PathLike, rename=True
) -> Catalog:
    """Build a catalog from a yaml file.

    Args:
        path_yaml (str | PathLike): path to the yaml file

    Returns:
        Catalog: a catalog instance with the data from the yaml file
    """

    yaml_filepath = DATA_BASE_PATH / yaml_filename
    with open(yaml_filepath, "r") as file:
        data = yaml.safe_load(file)

    if rename:
        rename_map = {
            key: value
            for list_item in data["columns_mapping"]
            for (key, value) in list_item.items()
        }
    else:
        rename_map = {}
    return Catalog(data["csv_path"], data["key_column_name"], rename_map)


fake_catalog = Catalog("pokemon.csv", key_column_name="Name")
iris_catalog = Catalog("iris_dataset.csv", key_column_name="sepal length (cm)")
sample_cable_catalog = build_catalog_from_yaml("sample_cable_database.yaml")
