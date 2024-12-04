from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel, RootModel


class ElementArray(ABC):
    def __init__(self, data: pd.DataFrame) -> None:
        input_data = data.copy()
        self.data = self.validate_input_data(input_data)

    @classmethod
    @abstractmethod
    def validate_input_data(cls, input_data: pd.DataFrame) -> pd.DataFrame:
        pass

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
    """Description of an overhead line section"""

    @classmethod
    def validate_input_data(cls, input_data) -> pd.DataFrame:
        columns = [
            "name",
            "suspension",
            "conductor_attachment_altitude",
            "crossarm_length",
            "line_angle",
            "insulator_length",
            "span_length",
        ]
        extra_columns = [
            column for column in input_data.columns if column not in columns
        ]
        data = input_data.drop(columns=extra_columns)
        records = data.to_dict("records")
        SectionInputData.model_validate(records)
        return data
