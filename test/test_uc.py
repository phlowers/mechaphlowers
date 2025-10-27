import mechaphlowers as mph

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# load data 
from mechaphlowers.data.units import Q_
from mechaphlowers.plotting.plot import PlotLine
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.entities.arrays import (
    CableArray,
    SectionArray,
)

def test_plot_engine_v0_4_0(balance_engine_base_test) -> None:

    section_array = SectionArray(
        pd.DataFrame(
            {
                "name": ["1", "2", "3", "4"],
                "suspension": [False, True, True, False],
                "conductor_attachment_altitude": [50, 100, 50, 50],
                "crossarm_length": [10, 10, 10, 10],
                "line_angle": Q_(np.array([0, 0, 0, 0]), "grad")
                .to('deg')
                .magnitude,
                "insulator_length": [3, 3, 3, 3],
                "span_length": [500, 500, 500, np.nan],
                "insulator_weight": [1000, 500, 500, 1000],
                "load_weight": [0, 0, 0, 0],
                "load_position": [0, 0, 0, 0],
            }
        )
    )
    section_array.sagging_parameter = 2000
    section_array.sagging_temperature = 15

    from mechaphlowers.data.catalog import sample_cable_catalog

    cable_array_AM600 = sample_cable_catalog.get_as_object(["ASTER600"])

    engine = BalanceEngine(cable_array=cable_array_AM600, section_array=section_array)

    engine = balance_engine_base_test

    plt_line = PlotLine.builder_from_balance_engine(engine)
    engine.solve_adjustment()
    engine.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1])
    )
    fig = go.Figure()
    plt_line.preview_line3d(fig)
    # engine.solve_change_state(
    #     wind_pressure=100 * np.array([1, 1, 1, np.nan]),
    # )
    engine.solve_change_state(
        new_temperature=45 * np.array([1, 1, 1])
    )
    # fig = go.Figure()
    plt_line.preview_line3d(fig)

    # assert (
    #     len(
    #         [
    #             f
    #             for f in fig.data
    #             if f.name == "Cable" and not np.isnan(f.x).all()  # type: ignore[attr-defined]
    #         ]
    #     )
    #     == 2
    # )

    fig.show() 
    
    plt_line = PlotLine.builder_from_balance_engine(balance_engine_base_test)
    balance_engine_base_test.solve_adjustment()
    balance_engine_base_test.solve_change_state(
        new_temperature=15 * np.array([1, 1, 1])
    )
    fig = go.Figure()
    plt_line.preview_line3d(fig)
    balance_engine_base_test.solve_change_state(
        wind_pressure=1000 * np.array([1, 1, 1, np.nan]),
    )

    plt_line.preview_line3d(fig)
    
    fig.show() 