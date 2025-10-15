# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.config import options as cfg
from mechaphlowers.core.models.cable.span import ISpan
from mechaphlowers.entities.arrays import CableArray

IMAGINARY_THRESHOLD = cfg.solver.deformation_imag_thresh  # type: ignore


class IDeformation(ABC):
    """This abstract class is a base class for models to compute relative cable deformations."""

    def __init__(
        self,
        tension_mean: np.ndarray,
        cable_length: np.ndarray,
        cable_section_area: np.float64,
        linear_weight: np.float64,
        young_modulus: np.float64,
        dilatation_coefficient: np.float64,
        temperature_reference: np.float64,
        polynomial_conductor: Poly,
        sagging_temperature: np.ndarray,
        max_stress: np.ndarray | None = None,
        **_,
    ):
        self.tension_mean = tension_mean
        self.cable_length = cable_length
        self.cable_section_area = cable_section_area
        self.linear_weight = linear_weight
        self.young_modulus = young_modulus
        self.dilatation_coefficient = dilatation_coefficient
        self.temp_ref = temperature_reference
        self.polynomial_conductor = polynomial_conductor
        self.current_temperature = sagging_temperature
        self.is_polynomial = polynomial_conductor.trim().degree() >= 2

        if max_stress is None:
            self.max_stress = np.full(self.cable_length.shape, 0)

    @abstractmethod
    def L_ref(self) -> np.ndarray:
        """Unstressed cable length, at a chosen reference temperature, compared to the temperature reference"""

    @abstractmethod
    def L_0(self) -> np.ndarray:
        """Unstressed cable length, at a chosen reference temperature, whrer temperature_reference = 0°C"""

    @abstractmethod
    def epsilon(self) -> np.ndarray:
        """Total relative strain of the cable."""

    @abstractmethod
    def epsilon_mecha(self) -> np.ndarray:
        """Mechanical part of the relative strain  of the cable."""

    @abstractmethod
    def epsilon_therm(self) -> np.ndarray:
        """Thermal part of the relative deformation of the cable, compared to a temperature_reference."""

    @abstractmethod
    def epsilon_therm_0(self) -> np.ndarray:
        """Thermal part of the relative deformation of the cable, where temperature_reference = 0."""


class DeformationRte(IDeformation):
    """This class implements the deformation model used by RTE."""

    def L_ref(self) -> np.ndarray:
        L = self.cable_length
        epsilon = self.epsilon_therm() + self.epsilon_mecha()
        return L / (1 + epsilon)

    def L_0(self) -> np.ndarray:
        L = self.cable_length
        epsilon = self.epsilon_therm_0() + self.epsilon_mecha()
        return L / (1 + epsilon)

    def epsilon_mecha(self) -> np.ndarray:
        T_mean = self.tension_mean
        E = self.young_modulus
        S = self.cable_section_area
        # linear case
        if not self.is_polynomial:
            return T_mean / (E * S)
        # polynomial case
        else:
            return self.epsilon_mecha_polynomial()

    def epsilon(self):
        return self.epsilon_mecha() + self.epsilon_therm()

    def epsilon_therm(self) -> np.ndarray:
        sagging_temperature = self.current_temperature
        temp_ref = self.temp_ref
        alpha = self.dilatation_coefficient
        return (sagging_temperature - temp_ref) * alpha

    def epsilon_therm_0(self) -> np.ndarray:
        sagging_temperature = self.current_temperature
        alpha = self.dilatation_coefficient
        return sagging_temperature * alpha

    def epsilon_mecha_polynomial(self) -> np.ndarray:
        """Computes epsilon when the stress-strain relation is polynomial"""
        T_mean = self.tension_mean
        E = self.young_modulus
        S = self.cable_section_area

        sigma = T_mean / S
        if self.polynomial_conductor is None:
            raise ValueError("Polynomial is not defined")
        epsilon_plastic = self.epsilon_plastic()
        return epsilon_plastic + sigma / E

    def epsilon_plastic(self) -> np.ndarray:
        """Computes elastic permanent strain."""
        T_mean = self.tension_mean
        E = self.young_modulus
        S = self.cable_section_area
        max_stress = self.max_stress

        sigma = T_mean / S
        if max_stress is None:
            max_stress = np.full(T_mean.shape, 0)
        # epsilon plastic is based on the highest value between sigma and max_stress
        highest_constraint = np.fmax(sigma, max_stress)
        equation_solution = self.resolve_stress_strain_equation(
            highest_constraint
        )
        equation_solution -= highest_constraint / E
        return equation_solution

    def resolve_stress_strain_equation(
        self, highest_constraint: np.ndarray
    ) -> np.ndarray:
        """Solves $\\sigma = Polynomial(\\varepsilon)$"""
        polynomial = self.polynomial_conductor

        polynom_array = np.full(highest_constraint.shape, polynomial)
        poly_to_resolve = polynom_array - highest_constraint
        return self.find_smallest_real_positive_root(poly_to_resolve)

    def find_smallest_real_positive_root(
        self,
        poly_to_resolve: np.ndarray,
    ) -> np.ndarray:
        """Find the smallest root that is real and positive for each polynomial

        Args:
                poly_to_resolve (np.ndarray): array of polynomials to solve

        Raises:
                ValueError: if no real positive root has been found for at least one polynomial.

        Returns:
                np.ndarray: array of the roots (one per polynomial)
        """
        # Can cause performance issues
        all_roots = [poly.roots() for poly in poly_to_resolve]

        all_roots_stacked = np.stack(all_roots)
        keep_solution_condition = np.logical_and(
            abs(all_roots_stacked.imag) < IMAGINARY_THRESHOLD,
            0.0 <= all_roots_stacked,
        )
        # Replace roots that are not real nor positive by np.inf
        real_positive_roots = np.where(
            keep_solution_condition, all_roots_stacked, np.inf
        )
        real_smallest_root = real_positive_roots.min(axis=1).real
        if np.inf in real_smallest_root:
            raise ValueError("No solution found for at least one span")
        return real_smallest_root


def deformation_model_builder(
    cable_array: CableArray,
    span_model: ISpan,
    sagging_temperature: np.ndarray,
    deformation_model_type: Type[IDeformation] = DeformationRte,
) -> IDeformation:
    tension_mean = span_model.T_mean()
    cable_length = span_model.compute_L()
    cable_section = np.float64(cable_array.data.section.iloc[0])
    linear_weight = np.float64(cable_array.data.linear_weight.iloc[0])
    young_modulus = np.float64(cable_array.data.young_modulus.iloc[0])
    dilatation_coefficient = np.float64(
        cable_array.data.dilatation_coefficient.iloc[0]
    )
    temperature_reference = np.float64(
        cable_array.data.temperature_reference.iloc[0]
    )
    polynomial_conductor = cable_array.polynomial_conductor
    return deformation_model_type(
        tension_mean,
        cable_length,
        cable_section,
        linear_weight,
        young_modulus,
        dilatation_coefficient,
        temperature_reference,
        polynomial_conductor,
        sagging_temperature,
    )
