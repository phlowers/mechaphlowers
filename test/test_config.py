# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import pytest

try:
    import plotly.graph_objects as go  # type: ignore[import-untyped]
except ImportError:
    pytest.skip("plotly is not installed", allow_module_level=True)

from mechaphlowers.api.frames import SectionDataFrame
from mechaphlowers.config import options
from mechaphlowers.entities.arrays import SectionArray


def test_config_on_plot(
    default_section_array_three_spans: SectionArray,
) -> None:
    frame = SectionDataFrame(default_section_array_three_spans)
    fig = go.Figure()
    options.graphics.resolution = 20
    options.graphics.marker_size = 10.0
    frame.plot.line3d(fig)
    assert (
        fig._data[0].get('marker').get('size') == options.graphics.marker_size  # type: ignore[attr-defined]
    )
    assert fig._data[0].get('x').shape[0] == (  # type: ignore[attr-defined]
        options.graphics.resolution + 1
    ) * (default_section_array_three_spans.data.shape[0] - 1)
    # fig.show() # deactivate for auto unit testing
