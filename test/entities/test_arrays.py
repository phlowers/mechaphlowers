import numpy as np
import pandera as pa
import pytest
from pandas.testing import assert_frame_equal
from pandera.typing import DataFrame

from mechaphlowers.entities.arrays import SectionArray, SectionInputDataFrame


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
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)
    section = SectionArray(input_df)

    assert_frame_equal(input_df, section.data, check_dtype=False, rtol=1e-07)


def test_create_section_array__missing_name(section_array_input_data: dict) -> None:
    del section_array_input_data["name"]
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)


def test_create_section_array__missing_suspension(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["suspension"]
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)


def test_create_section_array__missing_conductor_attachment_altitude(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["conductor_attachment_altitude"]
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)


def test_create_section_array__missing_crossarm_length(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["crossarm_length"]
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)


def test_create_section_array__missing_line_angle(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["line_angle"]
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)


def test_create_section_array__missing_insulator_length(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["insulator_length"]
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)


def test_create_section_array__missing_span_length(
    section_array_input_data: dict,
) -> None:
    del section_array_input_data["span_length"]
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)


def test_create_section_array__extra_column(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["extra column"] = [0] * 4

    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    section = SectionArray(input_df)

    assert "extra column" not in section.data.columns


def test_create_section_array__wrong_name_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["name"] = [1, 2, 3, 4]
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)


def test_create_section_array__wrong_suspension_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["suspension"] = [1, 2, 3, 4]
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)


def test_create_section_array__wrong_conductor_attachment_altitude_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["conductor_attachment_altitude"] = ["1,2"] * 4
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)


def test_create_section_array__wrong_crossarm_length_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["crossarm_length"] = ["1,2"] * 4
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)


def test_create_section_array__wrong_line_angle_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["line_angle"] = ["1,2"] * 4
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)


def test_create_section_array__wrong_insulator_length_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["insulator_length"] = ["1,2"] * 4
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)


def test_create_section_array__wrong_span_length_type(
    section_array_input_data: dict,
) -> None:
    section_array_input_data["span_length"] = ["1,2"] * 4
    input_df: DataFrame[SectionInputDataFrame] = DataFrame(section_array_input_data)

    with pytest.raises(pa.errors.SchemaErrors):
        SectionArray(input_df)
