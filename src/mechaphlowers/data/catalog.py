# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from pathlib import Path

import pandas as pd

# Resolve the 'data' folder
# which is the parent folder of this script
# in order to later be able to find the data files stored in this folder.
DATA_BASE_PATH = Path(__file__).absolute().parent


class Catalog:
	def __init__(self, filename, key_column_name) -> None:
		filepath = DATA_BASE_PATH / Path(filename)
		self._data = pd.read_csv(filepath, index_col=key_column_name)

	def get(self, keys: list) -> pd.DataFrame:
		return self._data.loc[keys]


fake_catalog = Catalog("pokemon.csv", key_column_name="Name")
iris_catalog = Catalog("iris_dataset.csv", key_column_name="sepal length (cm)")
