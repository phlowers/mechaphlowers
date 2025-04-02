# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import logging
from importlib.metadata import version

import pandas as pd

from mechaphlowers.api.frames import SectionDataFrame
from mechaphlowers.config import options

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

pd.options.mode.copy_on_write = True

__version__ = version('mechaphlowers')

logger.info("Mechaphlowers package initialized.")
logger.info(f"Mechaphlowers version: {__version__}")


__all__ = ["SectionDataFrame", "options"]
