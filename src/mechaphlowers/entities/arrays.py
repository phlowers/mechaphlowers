from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel, RootModel


class ElementArray(ABC):
    def __init__(self, data: pd.DataFrame) -> None:
        data = self.drop_extra_columns(data)
        self.check_data(data)
        self.data = data

    @property
    @abstractmethod
    def _input_columns(self) -> list[str]: ...

    @property
    @abstractmethod
    def _input_data_model(self): ...

    def _compute_extra_columns(self, input_data: pd.DataFrame) -> list[str]:
        return [
            column for column in input_data.columns if column not in self._input_columns
        ]

    def drop_extra_columns(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the input DataFrame, without irrelevant columns.

        Note: This doesn't change the input DataFrame.
        """
        extra_columns = self._compute_extra_columns(input_data)
        return input_data.drop(columns=extra_columns)

    def check_data(self, data: pd.DataFrame) -> None:
        records = data.to_dict("records")
        self._input_data_model.model_validate(records)

    def __str__(self) -> str:
        return self.data.to_string()


class SectionInputRecord(BaseModel):
    name: str
    suspension: bool
    conductor_attachment_altitude: float
    crossarm_length: float
    line_angle: float
    insulator_length: float
    span_length: float


class SectionInputData(RootModel):
    root: list[SectionInputRecord]


class SectionArray(ElementArray):
    """Description of an overhead line section."""

    @property
    def _input_data_model(self):
        return SectionInputData

    @property
    def _input_columns(self) -> list[str]:
        return [
            "name",
            "suspension",
            "conductor_attachment_altitude",
            "crossarm_length",
            "line_angle",
            "insulator_length",
            "span_length",
        ]
