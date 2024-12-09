from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series


class ElementArray(ABC):
    def __init__(self, data: DataFrame) -> None:
        data = self._drop_extra_columns(data)
        self.data = data

    @property
    @abstractmethod
    def _input_columns(self) -> list[str]: ...

    def _compute_extra_columns(self, input_data: DataFrame) -> list[str]:
        return [
            column for column in input_data.columns if column not in self._input_columns
        ]

    def _drop_extra_columns(self, input_data: DataFrame) -> DataFrame:
        """Return a copy of the input DataFrame, without irrelevant columns.

        Note: This doesn't change the input DataFrame.
        """
        extra_columns = self._compute_extra_columns(input_data)
        return input_data.drop(columns=extra_columns)

    def __str__(self) -> str:
        return self.data.to_string()


class SectionInputDataFrame(pa.DataFrameModel):  # TODO: rename
    name: Series[str]
    suspension: Series[bool]
    conductor_attachment_altitude: Series[float]
    crossarm_length: Series[float]
    line_angle: Series[float]
    insulator_length: Series[float]
    span_length: Series[float] = pa.Field(nullable=True)

    @pa.dataframe_check
    def insulators_only_for_suspension_supports(cls, df: DataFrame) -> Series[bool]:
        # Though this doesn't reflect reality,
        # for now we don't take into account insulator lengths
        # for the tension supports so we don't want to mislead the users
        # by accepting input data with positive insulator length
        # for tension supports.
        # Calculation with insulator lengths for tension supports
        # might be implemented later.
        return (df["suspension"] | (df["insulator_length"] == 0)).pipe(Series[bool])

    @pa.dataframe_check
    def no_span_length_for_last_row(cls, df: DataFrame) -> bool:
        # TODO: more explicit error message?
        return df.tail(1)["span_length"].isna().all()


class SectionArray(ElementArray):
    """Description of one or several overhead line sections."""

    @pa.check_types(lazy=True)
    def __init__(
        self,
        data: DataFrame[SectionInputDataFrame],
        sagging_parameter: float,
        sagging_temperature: float,
    ) -> None:
        super().__init__(data)
        self.sagging_parameter = sagging_parameter
        self.sagging_temperature = sagging_temperature

    @property
    def _input_columns(self) -> list[str]:
        metadata = SectionInputDataFrame.get_metadata()
        return metadata["SectionInputDataFrame"]["columns"].keys()  # type: ignore

    def export_data(self) -> pd.DataFrame:
        return self.data.assign(
            elevation_difference=self.compute_elevation_difference(),
            sagging_parameter=self.sagging_parameter,
            sagging_temperature=self.sagging_temperature,
        )

    def compute_elevation_difference(self) -> np.ndarray:
        left_support_height = self.data["conductor_attachment_altitude"]
        right_support_height = self.data["conductor_attachment_altitude"].shift(
            periods=-1
        )
        return (left_support_height - right_support_height).to_numpy()
