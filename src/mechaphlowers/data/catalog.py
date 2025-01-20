# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import pandas as pd


class FakeCatalog:
	def __init__(self) -> None:
		filepath = "src/mechaphlowers/data/pokemon.csv"
		key_column_name = "Name"
		self._data = pd.read_csv(filepath, index_col=key_column_name)

	def get(self, keys: list[str]) -> pd.DataFrame:
		return self._data.loc[keys]


fake_catalog = FakeCatalog()
