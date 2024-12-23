# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from mechaphlowers.entities.arrays import SectionArray
from mechaphlowers.entities import SectionFrame


data = {
    "name": ["support 1", "2", "three", "support 4"],
    "suspension": [False, True, True, False],
    "conductor_attachment_altitude": [50.0, 40.0, 20.0, 10.0],
    "crossarm_length": [5.0,]*4,
    "line_angle": [0,]*4,
    "insulator_length": [0, 4, 3.2, 0],
    "span_length": [100, 200, 300, np.nan],
    }

section = SectionArray(data = pd.DataFrame(data))
section.sagging_parameter = 500

frame = SectionFrame(section)


def test_plot_line3d():
    fig = go.Figure()
    frame.plot.line3d(fig)
    fig.show()
    assert True