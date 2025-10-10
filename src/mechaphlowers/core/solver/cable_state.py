# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from typing import Type

import numpy as np
from numpy.polynomial import Polynomial as Poly

try:
    from scipy import optimize  # type: ignore
except ImportError:
    import mechaphlowers.numeric.scipy as optimize

from mechaphlowers.config import options as cfg
from mechaphlowers.core.models.cable.deformation import (
    DeformationRte,
    IDeformation,
)
from mechaphlowers.core.models.cable.span import (
    CatenarySpan,
    ISpan,
)
from mechaphlowers.core.models.external_loads import CableLoads


class SagTensionSolver:
    """This class reprensents the sag tension calculation.
    It computes the horizontal tension of a cable after a change of parameters
    """

    _ZETA = cfg.solver.sagtension_zeta  # type: ignore

    def __init__(
        self,
        span_length: np.ndarray,
        elevation_difference: np.ndarray,
        sagging_parameter: np.ndarray,
        sagging_temperature: np.ndarray,
        cable_section_area: np.float64,
        diameter: np.float64,
        linear_weight: np.float64,
        young_modulus: np.float64,
        dilatation_coefficient: np.float64,
        temperature_reference: np.float64,
        polynomial_conductor: Poly,
        **_,
    ) -> None:
        self.span_length = span_length
        self.elevation_difference = elevation_difference
        self.sagging_parameter = sagging_parameter
        self.sagging_temperature = sagging_temperature
        self.cable_section_area = cable_section_area
        self.diameter = diameter
        self.linear_weight = linear_weight
        self.young_modulus = young_modulus
        self.dilatation_coefficient = dilatation_coefficient
        self.temperature_reference = temperature_reference
        self.polynomial_conductor = polynomial_conductor
        self.L_ref: np.ndarray
        self.span_model_type: Type[ISpan] = CatenarySpan
        self.deformation_model_type: Type[IDeformation] = DeformationRte
        self.T_h_after_change: np.ndarray | None = None
        self.initialize_cable_loads()
        # lengths in the new cable plane after considering wind pressure
        self.span_length_after_loads = span_length
        self.elevation_difference_after_loads = elevation_difference

    def initialize_cable_loads(self):
        "Initialize a cable_loads with no weather"
        self.cable_loads = CableLoads(
            self.diameter,
            self.linear_weight,
            np.zeros(self.span_length.shape),
            np.zeros(self.span_length.shape),
        )

    def initial_state(self) -> None:
        """Method that computes the unstressed length and the initial horizontal tension of the cable without any external loads.
        Store those two values in the class attributes.
        It should be called after the initialization of the class.

        """
        self.initialize_cable_loads()
        self.span_model = self.span_model_type(
            self.span_length,
            self.elevation_difference,
            self.sagging_parameter,
            # load coefficient used comes from the default CableLoads: should be equal to 1
            load_coefficient=self.cable_loads.load_coefficient,
            linear_weight=self.linear_weight,
        )
        cable_length = self.span_model.compute_L()
        tension_mean = self.span_model.T_mean()
        self.deformation_model = self.deformation_model_type(
            **self.__dict__,
            tension_mean=tension_mean,
            cable_length=cable_length,
        )
        self.L_ref = self.deformation_model.L_ref()
        self.T_h_after_change = self.span_model.T_h()

    def update_loads(
        self, ice_thickness: np.ndarray, wind_pressure: np.ndarray
    ) -> None:
        """Update CableLoads data with new ice_thickness and wind_pressure given in input.
        Also update span_length_after_loads and elevation_difference_after_loads that may change because of the wind

        Args:
                ice_thickness (np.ndarray): new ice thickness
                wind_pressure (np.ndarray): new wind pressure
        """
        self.cable_loads.ice_thickness = ice_thickness
        self.cable_loads.wind_pressure = wind_pressure
        beta = self.cable_loads.load_angle
        a = self.span_length
        b = self.elevation_difference
        self.span_length_after_loads = np.sqrt(a**2 + (b * np.sin(beta)) ** 2)
        self.elevation_difference_after_loads = b * np.cos(beta)

    def update_models_before_solving(self, new_temperature):
        self.span_model.load_coefficient = self.cable_loads.load_coefficient
        self.span_model.span_length = self.span_length_after_loads
        self.span_model.elevation_difference = (
            self.elevation_difference_after_loads
        )
        self.deformation_model.current_temperature = new_temperature

    def change_state(
        self,
        ice_thickness: np.ndarray,
        wind_pressure: np.ndarray,
        new_temperature: np.ndarray,
        solver: str = "newton",
        **_,
    ) -> None:
        """Method that solves the finds the new horizontal tension after a change of weather. Parameters are given in SI units.
        The equation to solve is : $\\delta(T_h) = O$
        Args:
                ice_thickness (np.ndarray): new ice_thickness (in m)
                wind_pressure (np.ndarray): new wind_pressure (in Pa)
                new_temperature (np.ndarray): new temperature
                solver (str, optional): resolution method of the equation. Defaults to "newton", which is the only method implemented for now.
        """

        solver_dict = {"newton": optimize.newton}
        try:
            solver_method = solver_dict[solver]
        except KeyError:
            raise ValueError(f"Incorrect solver name: {solver}")
        self.update_loads(ice_thickness, wind_pressure)
        self.update_models_before_solving(new_temperature)

        T_h0 = self.span_model.T_h()

        # TODO adapt code to use other solving methods

        solver_result = solver_method(
            self._delta,
            T_h0,
            fprime=self._delta_prime,
            tol=1e-5,
            full_output=True,
        )
        if not solver_result.converged.all():
            raise ValueError("Solver did not converge")
        self.T_h_after_change = solver_result.root

    def _delta(
        self,
        T_h: np.ndarray,
    ) -> np.ndarray:
        """Function to solve.
        This function is the difference between two ways to compute epsilon.
        Therefore, its value should be zero.
        $\\delta = \\varepsilon_{L} - \\varepsilon_{T}$
        """
        k_load = self.cable_loads.load_coefficient
        p = T_h / (k_load * self.linear_weight)

        self.span_model.set_parameter(p)
        L = self.span_model.compute_L()
        self.deformation_model.cable_length = L
        self.deformation_model.tension_mean = self.span_model.T_mean()

        epsilon_total = self.deformation_model.epsilon()
        return (L / self.L_ref - 1) - epsilon_total

    def _delta_prime(
        self,
        Th: np.ndarray,
    ) -> np.ndarray:
        """Approximation of the derivative of the function to solve
        $$\\delta'(T_h) = \\frac{\\delta(T_h + \\zeta) - \\delta(T_h)}{\\zeta}$$
        """
        return (self._delta(Th + self._ZETA) - self._delta(Th)) / self._ZETA

    def p_after_change(self) -> np.ndarray:
        """Compute the new value of the sagging parameter after sag tension calculation"""
        if self.T_h_after_change is None:
            raise ValueError(
                "initial_state() or change_state() has to be run before calling this method"
            )
        return self.span_model.sagging_parameter

    def L_after_change(self) -> np.ndarray:
        """Compute the new value of the length of the cable after sag tension calculation"""
        if self.T_h_after_change is None:
            raise ValueError(
                "initial_state() or change_state() has to be run before calling this method"
            )
        return self.span_model.compute_L()
