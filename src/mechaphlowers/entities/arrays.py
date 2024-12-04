from abc import ABC, abstractmethod

import pandas as pd
from pydantic import BaseModel, RootModel


class ElementArray(ABC):
    def __init__(self, data: pd.DataFrame) -> None:
        input_data = data.copy()
        data = self.drop_extra_columns(input_data)
        self.validate_input_data(data)
        self.data = data

    @staticmethod
    @abstractmethod
    def get_input_columns() -> list[str]: ...

    @staticmethod
    @abstractmethod
    def get_input_data_model(): ...

    @classmethod
    def compute_extra_columns(cls, input_data: pd.DataFrame) -> list[str]:
        return [
            column
            for column in input_data.columns
            if column not in cls.get_input_columns()
        ]

    @classmethod
    def drop_extra_columns(cls, input_data) -> pd.DataFrame:
        extra_columns = cls.compute_extra_columns(input_data)
        return input_data.drop(columns=extra_columns)

    @classmethod
    def validate_input_data(cls, data):
        records = data.to_dict("records")
        cls.get_input_data_model().model_validate(records)

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

    @staticmethod
    def get_input_columns():
        return [
            "name",
            "suspension",
            "conductor_attachment_altitude",
            "crossarm_length",
            "line_angle",
            "insulator_length",
            "span_length",
        ]

    @staticmethod
    def get_input_data_model():
        return SectionInputData
