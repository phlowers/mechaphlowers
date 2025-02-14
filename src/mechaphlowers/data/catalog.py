# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from os import PathLike
from pathlib import Path

import pandas as pd

from mechaphlowers.entities.arrays import CableArray

# Resolve the 'data' folder
# which is the parent folder of this script
# in order to later be able to find the data files stored in this folder.
DATA_BASE_PATH = Path(__file__).absolute().parent


class Catalog:
	"""Generic wrapper for tabular data read from a csv file, indexed by a `key` column."""

	def __init__(self, filename: str | PathLike, key_column_name: str) -> None:
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

	def get(self, keys: list) -> pd.DataFrame:
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
		try:
			return self._data.loc[keys]
		except KeyError as e:
			raise KeyError(
				f"Error when requesting catalog: {e.args[0]}"
			) from e

	def get_as_cable_array(
		self, keys: list
	) -> CableArray:  # FIXME(ai-qui): re-use .get?
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
		try:
			df = self._data.loc[keys]
			return CableArray(df)  # TODO(ai-qui): test data validation
			# TODO(ai-qui): make this generic (CableArray vs. generic Catalog)
		except KeyError as e:
			raise KeyError(
				f"Error when requesting catalog: {e.args[0]}"
			) from e

	def __str__(self) -> str:
		return self._data.to_string()


fake_catalog = Catalog("pokemon.csv", key_column_name="Name")
iris_catalog = Catalog("iris_dataset.csv", key_column_name="sepal length (cm)")
sample_cable_catalog = Catalog(
	"sample_cable_database.csv", key_column_name="name"
)
# TODO(ai-qui): remove fake catalogs?
