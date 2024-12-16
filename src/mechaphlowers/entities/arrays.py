from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series


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


class SectionInputDataFrame(pa.DataFrameModel):
    """Schema for the data expected for a dataframe used to instantiate a SectionArray.

    Each row describes a support and the following span (except the last row which "only" describes the last support).

    Notes:
        Line angles are expressed in degrees.

        insulator_length should be zero for the first and last supports, since for now mechaphlowers
        ignores them when computing the state of a span or section.
        Taking them into account might be implemented later.

        span_length should be zero or numpy.nan for the last row.
    """

    name: Series[str]
    suspension: Series[bool]
    conductor_attachment_altitude: Series[float] = pa.Field(coerce=True)
    crossarm_length: Series[float] = pa.Field(coerce=True)
    line_angle: Series[float] = pa.Field(coerce=True)
    insulator_length: Series[float] = pa.Field(coerce=True)
    span_length: Series[float] = pa.Field(nullable=True, coerce=True)

    @pa.dataframe_check(
        description="""Though tension supports also have insulators,
        for now we ignore them when computing the state of a span or section.
        Taking them into account might be implemented later.
        For now, set the insulator length to 0 for tension supports to suppress this error."""
    )
    def insulator_length_is_zero_if_not_suspension(cls, df: DataFrame) -> Series[bool]:
        return (df["suspension"] | (df["insulator_length"] == 0)).pipe(Series[bool])

    @pa.dataframe_check(
        description="""Each row in the dataframe contains information about a support
        and the span next to it, except the last support which doesn't have a "next" span.
        So, specifying a span_length in the last row doesn't make any sense.
        Please set span_length to "not a number" (numpy.nan) to suppress this error.""",
    )
    def no_span_length_for_last_row(cls, df: DataFrame) -> bool:
        return df.tail(1)["span_length"].isin([0, np.nan]).all()


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
        data: DataFrame[SectionInputDataFrame],
        sagging_parameter: float | None = None,
        sagging_temperature: float | None = None,
    ) -> None:
        super().__init__(data)
        self.sagging_parameter = sagging_parameter
        self.sagging_temperature = sagging_temperature

    @property
    def _input_columns(self) -> list[str]:
        metadata = SectionInputDataFrame.get_metadata()
        return metadata["SectionInputDataFrame"]["columns"].keys()  # type: ignore

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
