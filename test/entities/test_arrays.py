import numpy as np
import pandas as pd
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
