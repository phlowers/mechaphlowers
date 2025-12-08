# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import pytest
from pytest import fixture

from mechaphlowers.config import options as options
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.data.catalog.catalog import sample_cable_catalog
from mechaphlowers.data.catalog.sample_section import (
    section_factory_sample_data,
)
from mechaphlowers.data.units import convert_weight_to_mass
from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
)
from mechaphlowers.entities.shapes import SupportShape
from mechaphlowers.plotting.plot import (
    PlotEngine,
    TraceProfile,
    figure_factory,
    plot_points_2d,
    plot_support_shape,
)


def test_plot_loads(cable_array_AM600: CableArray):
    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [30, 50, 60, 65],
                "crossarm_length": [0, 10, -10, 0],
                "line_angle": [0, 10, 0, 0],
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 300, 400, np.nan],
                "insulator_mass": convert_weight_to_mass(
                    [1000, 500, 500, 1000]
                ),
                "load_mass": [50, 100, 0, np.nan],
                "load_position": [0.2, 0.4, 0.6, np.nan],
            }
        )
    )
    section_array.add_units({"line_angle": "grad"})

    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    balance_engine_one_load = BalanceEngine(
        cable_array=cable_array_AM600,
        section_array=section_array,
    )

    plt_engine = PlotEngine.builder_from_balance_engine(
        balance_engine_one_load
    )
    balance_engine_one_load.solve_adjustment()
    balance_engine_one_load.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1, 1])
    )
    fig = go.Figure()

    plt_engine.preview_line3d(fig)