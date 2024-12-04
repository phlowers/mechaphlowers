import numpy as np
import pandas as pd
import pydantic
import pytest
from pandas.testing import assert_frame_equal

from mechaphlowers.entities.arrays import SectionArray


@pytest.fixture
def section_array_input_data() -> dict:
    return {
        "name": ["support 1", "2", "three", "support 4"],
        "suspension": [False, True, True, False],
        "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
        "crossarm_length": [10, 12.1, 10, 10.1],
        "line_angle": [0, 360, 90.1, -90.2],
        "insulator_length": [0, 4, 3.2, 0],
        "span_length": [1, 500.2, 500.05, np.nan],
    }


def test_create_section_array(section_array_input_data: dict) -> None:
    input_df = pd.DataFrame(section_array_input_data)
    section = SectionArray(input_df)

    assert_frame_equal(section.data, input_df, rtol=1e-07)


def test_create_section_array__missing_name(section_array_input_data: dict) -> None:
    del section_array_input_data["name"]
    input_df = pd.DataFrame(section_array_input_data)

    with pytest.raises(pydantic.ValidationError):
        SectionArray(input_df)


def test_create_section_array__missing_suspension(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["suspension"]
    input_df = pd.DataFrame(section_array_input_data)

    with pytest.raises(pydantic.ValidationError):
        SectionArray(input_df)


def test_create_section_array__missing_conductor_attachment_altitude(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["conductor_attachment_altitude"]
    input_df = pd.DataFrame(section_array_input_data)

    with pytest.raises(pydantic.ValidationError):
        SectionArray(input_df)


def test_create_section_array__missing_crossarm_length(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["crossarm_length"]
    input_df = pd.DataFrame(section_array_input_data)

    with pytest.raises(pydantic.ValidationError):
        SectionArray(input_df)


def test_create_section_array__missing_line_angle(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["line_angle"]
    input_df = pd.DataFrame(section_array_input_data)

    with pytest.raises(pydantic.ValidationError):
        SectionArray(input_df)


def test_create_section_array__missing_insulator_length(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["insulator_length"]
    input_df = pd.DataFrame(section_array_input_data)

    with pytest.raises(pydantic.ValidationError):
        SectionArray(input_df)


def test_create_section_array__missing_span_length(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["span_length"]
    input_df = pd.DataFrame(section_array_input_data)

    with pytest.raises(pydantic.ValidationError):
        SectionArray(input_df)


def test_create_section_array__extra_column(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["extra column"] = [0] * 4

    input_df = pd.DataFrame(section_array_input_data)

    section = SectionArray(input_df)

    assert "extra column" not in section.data.columns
