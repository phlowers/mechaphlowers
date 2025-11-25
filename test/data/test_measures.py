import numpy as np
import pandas as pd

from mechaphlowers.data.measures import PapotoParameterMeasure, param_15_deg
from mechaphlowers.entities.arrays import CableArray, SectionArray


def test_papoto_floats():
    a = 498.565922913587
    HL = 0.0
    VL = 97.4327311161033
    HR = 162.614599621714
    VR = 88.6907631859419
    H1 = 5.1134354937127
    V1 = 98.4518011880176
    H2 = 19.6314054626454
    V2 = 97.6289296721015
    H3 = 97.1475339907774
    V3 = 87.9335010245142

    papoto = PapotoParameterMeasure()
    papoto(
        a=a,
        HL=HL,
        VL=VL,
        HR=HR,
        VR=VR,
        H1=H1,
        V1=V1,
        H2=H2,
        V2=V2,
        H3=H3,
        V3=V3,
    )
    np.testing.assert_allclose(
        papoto.parameter, np.array([2000, np.nan]), atol=1.0
    )
    np.testing.assert_allclose(
        papoto.parameter_1_2, np.array([1999.78, np.nan]), atol=0.1
    )
    np.testing.assert_allclose(
        papoto.validity, np.array([8.85880213e-05, np.nan]), atol=1e-5
    )
    np.testing.assert_allclose(
        papoto.check_validity(), np.array([True, False]), atol=0.1
    )


def test_papoto_parameter_measure():
    a = np.array([498.565922913587, np.nan])
    HL = np.array([0.0, np.nan])
    VL = np.array([97.4327311161033, np.nan])
    HR = np.array([162.614599621714, np.nan])
    VR = np.array([88.6907631859419, np.nan])
    H1 = np.array([5.1134354937127, np.nan])
    V1 = np.array([98.4518011880176, np.nan])
    H2 = np.array([19.6314054626454, np.nan])
    V2 = np.array([97.6289296721015, np.nan])
    H3 = np.array([97.1475339907774, np.nan])
    V3 = np.array([87.9335010245142, np.nan])

    papoto = PapotoParameterMeasure()
    papoto(
        a=a,
        HL=HL,
        VL=VL,
        HR=HR,
        VR=VR,
        H1=H1,
        V1=V1,
        H2=H2,
        V2=V2,
        H3=H3,
        V3=V3,
    )

    np.testing.assert_allclose(
        papoto.parameter, np.array([2000, np.nan]), atol=1.0
    )
    np.testing.assert_allclose(
        papoto.parameter_1_2, np.array([1999.78, np.nan]), atol=0.1
    )
    np.testing.assert_allclose(
        papoto.validity, np.array([8.85880213e-05, np.nan]), atol=1e-5
    )
    np.testing.assert_allclose(
        papoto.check_validity(), np.array([True, False]), atol=0.1
    )


def test_select_points_in_dict():
    # Prepare mock data
    data = {
        "a": np.array([1]),
        "HL": np.array([2]),
        "VL": np.array([3]),
        "HR": np.array([4]),
        "VR": np.array([5]),
        "H1": np.array([10]),
        "V1": np.array([11]),
        "H2": np.array([20]),
        "V2": np.array([21]),
        "H3": np.array([30]),
        "V3": np.array([31]),
    }

    # Select points 1 and 3
    result = PapotoParameterMeasure.select_points_in_dict(1, 3, data)

    # Check that only H1/V1 and H3/V3 are present, and H2/V2 are replaced
    assert result["H1"].tolist() == [10]
    assert result["V1"].tolist() == [11]
    assert result["H2"].tolist() == [30]
    assert result["V2"].tolist() == [31]
    # Other keys should be present and unchanged
    assert result["a"].tolist() == [1]
    assert result["HL"].tolist() == [2]
    assert result["VL"].tolist() == [3]
    assert result["HR"].tolist() == [4]
    assert result["VR"].tolist() == [5]
    # H3/V3 should not be present as keys
    assert "H3" not in result
    assert "V3" not in result

    # Select points 2 and 1
    result2 = PapotoParameterMeasure.select_points_in_dict(2, 1, data)
    assert result2["H1"].tolist() == [20]
    assert result2["V1"].tolist() == [21]
    assert result2["H2"].tolist() == [10]
    assert result2["V2"].tolist() == [11]


def test_parameter_15_deg(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": [0, 10, 0, 0],
                "insulator_length": [0.001, 3, 3, 0.001],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": [00, 50, 50, 00],
                "load_mass": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    param_0 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=0
    )
    np.testing.assert_allclose(param_0, 2548.389, atol=1e-2)

    param_1 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=1
    )
    np.testing.assert_allclose(param_1, 2578.602, atol=1e-2)

    param_2 = param_15_deg(
        2000, 60, section_array, cable_array_AM600, span_index=2
    )
    np.testing.assert_allclose(param_2, 2576.706, atol=1e-2)
