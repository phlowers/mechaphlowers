# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import pytest
from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.entities import SectionFrame
from mechaphlowers.core.models.cable_models import CatenaryCableModel

import plotly.graph_objects as go


data = {
    "name": ["support 1", "2", "three", "support 4"],
    "suspension": [False, True, True, False],
    "conductor_attachment_altitude": [2.2, 5, -0.12, 0],
    "crossarm_length": [10, 12.1, 10, 10.1],
    "line_angle": [0, 360, 90.1, -90.2],
    "insulator_length": [0, 4, 3.2, 0],
    "span_length": [1, 500.2, 500., np.nan],
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
    
def test_select_spans__input():
    frame = SectionFrame(section)
    
    with pytest.raises(ValueError):
        frame.select(["support 1", "2", "three"])
        frame.select(["support 1",])
        
        frame.select(["support 1", "support 1"])
        frame.select(["support 1", "name_not_existing"])
        frame.select(["three", "support 1"])
        
    with pytest.raises(TypeError):
        frame.select("support 1")
        frame.select(["string", 2])
      
    frame_selected = frame.select(["support 1", "three"])
        
    assert len(frame_selected.data) == 3
    assert frame_selected.data.elevation_difference.take([1]).item() == frame.data.elevation_difference.take([1]).item()
    
    frame_selected = frame.select(["2", "support 4"])
    assert len(frame_selected.data) == 3
    assert frame_selected.data.elevation_difference.take([1]).item() == frame.data.elevation_difference.take([2]).item()
    
    
        
    

