# Copyright (c) 2024, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from mechaphlowers import plotting
from mechaphlowers.entities.frames import _SectionFrame as SectionFrame
from mechaphlowers.plotting.plot import plot_line
from mechaphlowers.utils import CachedAccessor

# adding plot module to SectionFrame object
SectionFrame.plot_line = plot_line
SectionFrame.plot = CachedAccessor("plot", plotting.plot.PlotAccessor)