# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Module for mechaphlowers configuration settings"""

from dataclasses import dataclass, field
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
class DataConfig:
    """configuration for data loading and saving"""

    sagging_temperature_default: float = 15.0


@dataclass
class GraphicsConfig:
    """Graphics configuration class."""

    resolution: int = 7
    marker_size: float = 3.0
    width: float = 8.0
    background_opacity: float = 0.3
    cable_trace_profile: dict = field(
        default_factory=lambda: {
            "name": "cable",
            "color": "dodgerblue",
        }
    )
    support_trace_profile: dict = field(
        default_factory=lambda: {
            "name": "support",
            "color": "indigo",
        }
    )
    insulator_trace_profile: dict = field(
        default_factory=lambda: {
            "name": "insulator",
            "color": "red",
            "size": 5.0,
        }
    )


@dataclass
class SolverConfig:
    """Solvers configuration class."""

    sagtension_zeta: float = 10.0
    deformation_imag_thresh: float = 1e-5
    balance_solver_change_state_params: dict = field(
        default_factory=lambda: {
            "perturb": 0.0001,
            "stop_condition": 1e-2,
            "relax_ratio": 0.8,
            "relax_power": 3,
            "max_iter": 100,
        }
    )
    balance_solver_adjustment_params: dict = field(
        default_factory=lambda: {
            "perturb": 0.0001,
            "stop_condition": 1e-2,
            "relax_ratio": 0.9,
            "relax_power": 1,
            "max_iter": 100,
        }
    )
    balance_solver_load_params: dict = field(
        default_factory=lambda: {
            "perturb": 0.001,
            "stop_condition": 1.0,
            "relax_ratio": 0.5,
            "relax_power": 3,
            "max_iter": 100,
        }
    )


@dataclass
class ComputeConfig:
    """ComputeConfig configuration class."""

    span_model: str = "CatenarySpan"
    deformation_model: str = "DeformationRte"


@dataclass
class GroundConfig:
    """Configuration class about ground."""

    default_support_length: float = 30.0


class LogConfig:
    """Logging configuration class."""

    perfs: bool = True


@dataclass
class InputUnitsConfig:
    cable_array: dict[str, str] = field(
        default_factory=lambda: {
            "section": "mm^2",
            "diameter": "mm",
            "young_modulus": "MPa",
            "linear_mass": "kg/m",
            "dilatation_coefficient": "1/K",
            "temperature_reference": "°C",
            "a0": "MPa",
            "a1": "MPa",
            "a2": "MPa",
            "a3": "MPa",
            "a4": "MPa",
            "b0": "MPa",
            "b1": "MPa",
            "b2": "MPa",
            "b3": "MPa",
            "b4": "MPa",
            "diameter_heart": "mm",
            "section_conductor": "mm^2",
            "section_heart": "mm^2",
            "electric_resistance_20": "ohm.m**-1",
            "linear_resistance_temperature_coef": "K**-1",
            "radial_thermal_conductivity": "W.m**-1.K**-1",
        }
    )
    section_array: dict[str, str] = field(
        default_factory=lambda: {
            "conductor_attachment_altitude": "m",
            "crossarm_length": "m",
            "line_angle": "grad",
            "insulator_length": "m",
            "span_length": "m",
            "insulator_mass": "kg",
        }
    )


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
        self._ground = GroundConfig()
        self._log = LogConfig()
        self._input_units = InputUnitsConfig()
        self._data_config = DataConfig()

    @property
    def ground(self) -> GroundConfig:
        """Ground configuration property."""
        return self._ground

    @property
    def output_units(self) -> OutputUnitsConfig:
        """Output units configuration property."""
        return self._output_units

    @property
    def input_units(self) -> InputUnitsConfig:
        """Input units configuration property."""
        return self._input_units

    @property
    def data(self) -> DataConfig:
        return self._data_config

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

    @property
    def log(self) -> LogConfig:
        """Logging configuration property."""
        return self._log

    class OptionError(Exception):
        """Exception raised when an option is not available."""

        def __init__(self, message: str):
            super().__init__(message)


# Declare below a ready to use options object
options = Config()
