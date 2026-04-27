# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from importlib.metadata import version

import pandas as pd

from mechaphlowers.api.section_study import SectionStudy
from mechaphlowers.config import options
from mechaphlowers.core.geometry.position_engine import PositionEngine
from mechaphlowers.core.models.balance.engine import BalanceEngine
from mechaphlowers.core.models.cable.thermal import ThermalEngine
from mechaphlowers.core.models.guying import Guying
from mechaphlowers.data.catalog.catalog import sample_cable_catalog
from mechaphlowers.data.geography.helpers import lambert93_to_gps
from mechaphlowers.data.measures import (
    PapotoParameterMeasure,
    param_calibration,
)
from mechaphlowers.data.units import Q_ as units
from mechaphlowers.entities.arrays import CableArray, SectionArray
from mechaphlowers.entities.geography import (
    GeoLocator,
    get_azimuth_from_gps,
    get_dist_and_angles_from_gps,
    get_dist_and_angles_from_lambert,
)
from mechaphlowers.entities.shapes import SupportShape
from mechaphlowers.plotting.plot import PlotEngine
from mechaphlowers.utils import ArrayTools

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

pd.options.mode.copy_on_write = True

__version__ = version('mechaphlowers')

logger.info("Mechaphlowers package initialized.")
logger.info(f"Mechaphlowers version: {__version__}")


__all__ = [
    "options",
    "SectionStudy",
    "BalanceEngine",
    "PlotEngine",
    "PositionEngine",
    "SectionArray",
    "CableArray",
    "SupportShape",
    "sample_cable_catalog",
    "units",
    "PapotoParameterMeasure",
    "param_calibration",
    "ThermalEngine",
    "Guying",
    "get_azimuth_from_gps",
    "get_dist_and_angles_from_gps",
    "get_dist_and_angles_from_lambert",
    "GeoLocator",
    "lambert93_to_gps",
    "ArrayTools",
]
