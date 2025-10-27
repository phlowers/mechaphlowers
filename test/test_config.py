# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import plotly.graph_objects as go  # type: ignore[import-untyped]

from mechaphlowers.config import options
from mechaphlowers.plotting.plot import PlotEngine


def test_config_on_plot(balance_engine_base_test) -> None:
    balance_engine_base_test.solve_adjustment()
    plt_line = PlotEngine.builder_from_balance_engine(balance_engine_base_test)

    fig = go.Figure()
    options.graphics.resolution = 20
    options.graphics.marker_size = 10.0
    plt_line.preview_line3d(fig)
    assert (
        fig._data[0].get('marker').get('size') == options.graphics.marker_size  # type: ignore[attr-defined]
    )
    assert fig._data[0].get('x').shape[0] == (  # type: ignore[attr-defined]
        options.graphics.resolution + 1
    ) * (balance_engine_base_test.section_array.data.shape[0] - 1)
    # fig.show() # deactivate for auto unit testing
