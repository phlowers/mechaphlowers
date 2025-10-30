# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Module for mechaphlowers configuration settings"""

from dataclasses import dataclass
from typing import Type

import numpy as np


@dataclass
class OutputUnitsConfig:
    """Units configuration class."""

    force: str = "daN"
    length: str = "m"
    mass: str = "kg"
    time: str = "s"
    temperature: str = "degC"


@dataclass
class PrecisionConfig:
    """Precision configuration class."""

    dtype_float: Type[np.floating] = np.float64
    dtype_int: Type[np.integer] = np.int64


@dataclass
class GraphicsConfig:
    """Graphics configuration class."""

    resolution: int = 7
    marker_size: float = 3.0


@dataclass
class SolverConfig:
    """Solvers configuration class."""

    sagtension_zeta: float = 10.0
    deformation_imag_thresh: float = 1e-5


@dataclass
class ComputeConfig:
    """ComputeConfig configuration class."""

    span_model: str = "CatenarySpan"
    deformation_model: str = "DeformationRte"


@dataclass
class Config:
    """Configuration class for mechaphlowers settings.

    This class is not intended to be used directly. Other classes
    are using the options instance to provide configuration settings. Default values are set in the class.
    `options` is available in the module mechaphlowers.config.

    Attributes:
            graphics_resolution (int): Resolution of the graphics.
            graphics_marker_size (float): Size of the markers in the graphics.
    """

    def __init__(self):
        self._graphics = GraphicsConfig()
        self._solver = SolverConfig()
        self._compute_config = ComputeConfig()
        self._precision = PrecisionConfig()
        self._output_units = OutputUnitsConfig()

    @property
    def output_units(self) -> OutputUnitsConfig:
        """Units configuration property."""
        return self._output_units

    @property
    def graphics(self) -> GraphicsConfig:
        """Graphics configuration property."""
        return self._graphics

    @property
    def solver(self) -> SolverConfig:
        """Solver configuration property."""
        return self._solver

    @property
    def compute(self) -> ComputeConfig:
        """Dataframe configuration property."""
        return self._compute_config

    @property
    def precision(self) -> PrecisionConfig:
        """Precision configuration property."""
        return self._precision

    class OptionError(Exception):
        """Exception raised when an option is not available."""

        def __init__(self, message: str):
            super().__init__(message)


# Declare below a ready to use options object
options = Config()
