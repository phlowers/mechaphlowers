from abc import ABC

import pandas as pd


class ElementArray(ABC):
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data.copy()

    def __str__(self) -> str:
        return self.data.to_string()


class SectionArray(ElementArray):
    """Description of an overhead line section"""
