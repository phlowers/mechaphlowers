# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from mechaphlowers.entities.schemas import SectionArrayInput


class ElementArray(ABC):
    def __init__(self, data: DataFrame) -> None:
        data = self._drop_extra_columns(data)
        self._data = data

    @property
    @abstractmethod
    def _input_columns(self) -> list[str]: ...

    def _compute_extra_columns(self, input_data: DataFrame) -> list[str]:
        return [
            column for column in input_data.columns if column not in self._input_columns
        ]

    def _drop_extra_columns(self, input_data: DataFrame) -> DataFrame:
        """Return a copy of the input DataFrame, without irrelevant columns.

        Note: This has no impact on the input DataFrame.
        """
        extra_columns = self._compute_extra_columns(input_data)
        return input_data.drop(columns=extra_columns)

    def __str__(self) -> str:
        return self._data.to_string()


class SectionArray(ElementArray):
    """Description of an overhead line section.

    Args:
        data: Input data
        sagging_parameter: Sagging parameter
        sagging_temperature: Sagging temperature, in Celsius degrees
    """

    @pa.check_types(lazy=True)
    def __init__(
        self,
        data: DataFrame[SectionArrayInput],
        sagging_parameter: float | None = None,
        sagging_temperature: float | None = None,
    ) -> None:
        super().__init__(data)
        self.sagging_parameter = sagging_parameter
        self.sagging_temperature = sagging_temperature

    @property
    def _input_columns(self) -> list[str]:
        metadata = SectionArrayInput.get_metadata()
        return metadata["SectionArrayInput"]["columns"].keys()  # type: ignore

    def compute_elevation_difference(self) -> np.ndarray:
        left_support_height = self._data["conductor_attachment_altitude"]
        right_support_height = self._data["conductor_attachment_altitude"].shift(
            periods=-1
        )
        return (left_support_height - right_support_height).to_numpy()

    @property
    def data(self) -> pd.DataFrame:
        return self._data.assign(
            elevation_difference=self.compute_elevation_difference(),
            sagging_parameter=self.sagging_parameter,
            sagging_temperature=self.sagging_temperature,
        )
