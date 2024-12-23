import numpy as np
import pandas as pd
from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.entities import SectionFrame
from mechaphlowers.core.models.cable_models import CatenaryCableModel


data = {
    "name": ["support 1", "2", "three", "support 4"],
    "suspension": [False, True, True, False],
    "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
    "crossarm_length": [10, 12.1, 10, 10.1],
    "line_angle": [0, 360, 90.1, -90.2],
    "insulator_length": [0, 4, 3.2, 0],
    "span_length": [1, 500.2, 500.05, np.nan],
    }

section = SectionArray(data = pd.DataFrame(data))
section.sagging_parameter = 2000

def test_section_frame_initialization():
    frame = SectionFrame(section)
    assert frame.section == section
    assert isinstance(frame.span_model, type(CatenaryCableModel))

def test_section_frame_get_coord():
    frame = SectionFrame(section)
    coords = frame.get_coord()
    assert coords.shape == (30,3)
    assert isinstance(coords, np.ndarray)
