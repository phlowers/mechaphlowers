from abc import ABC, abstractmethod

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
        return df.tail(1)["span_length"].isna().all()


class SectionArray(ElementArray):
    """Description of one or several overhead line sections."""

    @pa.check_types(lazy=True)
    def __init__(self, data: DataFrame[SectionInputDataFrame]) -> None:
        super().__init__(data)

    @property
    def _input_columns(self) -> list[str]:  # TODO: derive from SectionInputDataFrame?
        return [
            "name",
            "suspension",
            "conductor_attachment_altitude",
            "crossarm_length",
            "line_angle",
            "insulator_length",
            "span_length",
        ]
