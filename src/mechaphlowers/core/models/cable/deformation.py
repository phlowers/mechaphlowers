# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod

import numpy as np
from numpy.polynomial import Polynomial as Poly

from mechaphlowers.config import options as cfg
from mechaphlowers.entities.data_container import DataCable

IMAGINARY_THRESHOLD = cfg.solver.deformation_imag_thresh  # type: ignore


class SigmaFunctionSingleMaterial:
    def __init__(
        self,
        stress_strain_polynomial: Poly,
        young_modulus,
        dilatation_coefficient,
        T_labo,
    ) -> None:
        self.stress_strain_polynomial = stress_strain_polynomial
        self.young_modulus = young_modulus
        self.dilatation_coefficient = dilatation_coefficient
        self.T_labo = T_labo  # rename temp ref?
        self.sigma_max: np.ndarray | None = None
        self.epsilon_plastic_max: np.ndarray | None = None

    # TODO: best way to do this?
    def set_sigma_max(self, sigma_max: np.ndarray):
        self.epsilon_plastic_max = self.epsilon_plastic(
            sigma_max, 15 * np.ones_like(sigma_max)
        )  # TODO !!!!
        self.sigma_max = sigma_max

    def epsilon_total(self, sigma, current_temperature):
        E = self.young_modulus
        # TODO: unnecessary, but keep anyway?
        if E == np.float64(0.0):
            return np.zeros_like(sigma)
        eps_th = self.epsilon_th(current_temperature)
        eps_plast = self.epsilon_plastic(sigma, current_temperature)
        return eps_th + eps_plast + sigma / E

    def epsilon_th(self, current_temperature):
        return self.dilatation_coefficient * (
            current_temperature - self.T_labo
        )

    def epsilon_plastic(self, sigma, current_temperature):
        # TODO : do better than this
        if self.epsilon_plastic_max is None:
            self.epsilon_plastic_max = np.zeros_like(sigma)
        if self.sigma_max is None:
            self.sigma_max = np.zeros_like(sigma)
        need_recomputation = sigma > self.sigma_max
        sigma_recomputation = np.extract(need_recomputation, sigma)
        E = self.young_modulus
        if len(sigma_recomputation) > 0:
            new_epsilon_plastic_max = self.resolve_stress_strain_equation(
                sigma_recomputation, current_temperature
            ) - (sigma_recomputation / E)
            np.place(
                self.epsilon_plastic_max,
                need_recomputation,
                new_epsilon_plastic_max,
            )
        return self.epsilon_plastic_max

    def sigma_poly(self, x: np.ndarray):
        out = self.stress_strain_polynomial(x)
        return out

    def resolve_stress_strain_equation(
        self, sigma: np.ndarray, current_temperature: np.ndarray
    ) -> np.ndarray:
        """Solves $\\sigma = Polynomial(\\varepsilon_{plastic})$ for sigma values that need it"""
        # TODO: for loop here instead
        eps_th = self.epsilon_th(current_temperature)
        polynomials_array = np.full(sigma.shape, self.stress_strain_polynomial)
        polynomials_translated = [
            polynomials_array[index].convert(
                domain=[eps_th[index] - 1, eps_th[index] + 1]
            )
            for index in range(len(polynomials_array))
        ]
        poly_to_resolve = polynomials_translated - sigma
        return self.find_smallest_real_positive_root(poly_to_resolve)

    @staticmethod
    def find_smallest_real_positive_root(
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


class IDeformation(ABC):
    """This abstract class is a base class for models to compute relative cable deformations."""

    def __init__(
        self,
        data_cable: DataCable,
        tension_mean: np.ndarray,
        cable_length: np.ndarray,
        sagging_temperature: np.ndarray,
        # TODO: add setter max_stress to add it to SimpleMaterials?
        max_stress: np.ndarray | None = None,
        **kwargs,
    ):
        self.tension_mean = tension_mean
        self.cable_length = cable_length
        self.data_cable = data_cable
        self.current_temperature = sagging_temperature

        if max_stress is None:
            self._max_stress = np.full(self.cable_length.shape, 0.0)

    # # Keep this in abstract class?
    @property
    @abstractmethod
    def max_stress(self):
        """quoi"""
        pass

    @max_stress.setter
    @abstractmethod
    def max_stress(self, value_max_stress):
        """feur"""
        pass

    @abstractmethod
    def L_ref(self) -> np.ndarray:
        """Unstressed cable length, at a chosen reference temperature"""

    @abstractmethod
    def epsilon(self) -> np.ndarray:
        """Total relative strain of the cable."""

    # @abstractmethod
    # def epsilon_mecha(self) -> np.ndarray:
    #     """Mechanical part of the relative strain  of the cable."""

    # @abstractmethod
    # def epsilon_therm(self, current_temperature: np.ndarray) -> np.ndarray:
    #     """Thermal part of the relative deformation of the cable, compared to a temperature_reference."""

    @staticmethod
    @abstractmethod
    def compute_epsilon(
        T_mean: np.ndarray,
        data_cable: DataCable,
        current_temperature: np.ndarray,
        sigma_func_conductor: SigmaFunctionSingleMaterial,
        sigma_func_heart: SigmaFunctionSingleMaterial,
        max_stress: np.ndarray | None = None,
    ) -> np.ndarray:
        """Computing mechanical strain using a static method"""

    # @staticmethod
    # @abstractmethod
    # def compute_epsilon_therm(
    #     theta: np.ndarray, theta_ref: np.float64, alpha: np.float64
    # ) -> np.ndarray:
    #     """Computing thermal strain using a static method"""


class DeformationRte(IDeformation):
    """This class implements the deformation model used by RTE."""

    def __init__(
        self,
        data_cable: DataCable,
        tension_mean: np.ndarray,
        cable_length: np.ndarray,
        sagging_temperature: np.ndarray,
        max_stress: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(
            data_cable,
            tension_mean,
            cable_length,
            sagging_temperature,
            max_stress,
        )
        conductor_kwargs = {
            "stress_strain_polynomial": data_cable.polynomial_conductor,
            "young_modulus": data_cable.young_modulus_conductor,
            "dilatation_coefficient": data_cable.dilatation_coefficient_conductor,
            "T_labo": data_cable.temperature_reference,
        }
        heart_kwargs = {
            "stress_strain_polynomial": data_cable.polynomial_heart,
            "young_modulus": data_cable.young_modulus_heart,
            "dilatation_coefficient": data_cable.dilatation_coefficient_heart,
            "T_labo": data_cable.temperature_reference,
        }
        self.sigma_func_conductor = SigmaFunctionSingleMaterial(
            **conductor_kwargs
        )
        self.sigma_func_heart = SigmaFunctionSingleMaterial(**heart_kwargs)
        self.max_stress = self._max_stress

    @property
    def max_stress(self):
        return self._max_stress

    @max_stress.setter
    def max_stress(self, value_max_stress: np.ndarray):
        self.sigma_func_conductor.set_sigma_max(value_max_stress)
        if self.sigma_func_heart.young_modulus > np.float64(0):
            self.sigma_func_heart.set_sigma_max(value_max_stress)

    def L_ref(self) -> np.ndarray:
        L = self.cable_length
        epsilon = self.epsilon()
        return L / (1 + epsilon)

    def epsilon(self) -> np.ndarray:
        T_mean = self.tension_mean
        return self.compute_epsilon(
            T_mean,
            self.data_cable,
            self.current_temperature,
            self.sigma_func_conductor,
            self.sigma_func_heart,
            self.max_stress,
        )

    @staticmethod
    # epsilon total
    def compute_epsilon(
        T_mean: np.ndarray,
        data_cable: DataCable,
        current_temperature: np.ndarray,
        sigma_func_conductor: SigmaFunctionSingleMaterial,
        sigma_func_heart: SigmaFunctionSingleMaterial,
        max_stress: np.ndarray | None = None,
    ) -> np.ndarray:
        S = data_cable.cable_section_area
        sigma = T_mean / S  # global section?

        # single material
        eps_total_cond = sigma_func_conductor.epsilon_total(
            sigma, current_temperature
        )

        if sigma_func_heart.young_modulus == np.float64(0):
            # TODO: merge two cases?
            return eps_total_cond

        # two materials
        else:
            eps_plastic_cond = sigma_func_conductor.epsilon_plastic(
                sigma, current_temperature
            )
            eps_th_cond = sigma_func_conductor.epsilon_th(current_temperature)
            eps_total_heart = sigma_func_heart.epsilon_total(
                sigma, current_temperature
            )

            is_only_heart_supported: np.ndarray = (
                eps_total_heart < eps_plastic_cond + eps_th_cond
            )

            return np.where(
                is_only_heart_supported,
                eps_total_heart,
                eps_total_heart + eps_total_cond,
            )

    # def epsilon_therm(self, current_temperature: np.ndarray) -> np.ndarray:
    #     temp_ref = self.data_cable.temperature_reference
    #     alpha = self.data_cable.dilatation_coefficient
    #     return self.compute_epsilon_therm(current_temperature, temp_ref, alpha)

    # @staticmethod
    # # epsilon total
    # def compute_epsilon_mecha(
    #     T_mean: np.ndarray,
    #     data_cable: DataCable,
    #     max_stress: np.ndarray | None = None,
    # ) -> np.ndarray:
    #     # linear case
    #     if data_cable.polynomial_conductor.trim().degree() < 2:
    #         E = data_cable.young_modulus
    #         S = data_cable.cable_section_area
    #         return T_mean / (E * S)
    #     # add linear case with two materials

    #     # polynomial case
    #     # change way things work here?
    #     else:
    #         # test_data
    #         data_cable.young_modulus_heart
    #         data_cable.dilatation_coefficient
    #         data_cable.dilatation_coefficient_conductor
    #         data_cable.dilatation_coefficient_heart
    #         data_cable.cable_section_area_conductor
    #         return DeformationRte.compute_epsilon_mecha_polynomial(
    #             T_mean, data_cable, max_stress
    #         )

    # @staticmethod
    # def compute_epsilon_mecha_polynomial(
    #     T_mean: np.ndarray,
    #     data_cable: DataCable,
    #     max_stress: np.ndarray | None = None,
    # ) -> np.ndarray:
    #     """Computes epsilon when the stress-strain relation is polynomial"""
    #     S = data_cable.cable_section_area
    #     E = data_cable.young_modulus
    #     polynomial_conductor = data_cable.polynomial_conductor
    #     sigma = T_mean / S
    #     if polynomial_conductor is None:
    #         raise ValueError("Polynomial is not defined")
    #     # if sigma > sigma_max: sum of polynomials
    #     # else:
    #     # compute both eps_plastic + eps_th and compare them
    #     # take the lowest and compute total eps
    #     # compare total eps from first material to eps_pl + eps_th to other material

    #     # gérer les histoires de contrainte réduite

    #     # epsilon_plastic_conductor = DeformationRte.compute_epsilon_plastic(
    #     #     T_mean, (E - E_heart), S, polynomial_conductor, max_stress
    #     # )
    #     # epsilon_therm_conductor = DeformationRte.compute_epsilon_therm(theta, temp_ref, alpha_conductor)
    #     # epsilon_plastic_heart = DeformationRte.compute_epsilon_plastic(
    #     #     T_mean, E_heart, S, polynomial_heart, max_stress
    #     # )
    #     # epsilon_therm_heart = DeformationRte.compute_epsilon_therm(theta, temp_ref, alpha_conductor)
    #     # epsilon_total_heart = epsilon_plastic_conductor + epsilon_therm_heart + sigma / E
    #     # # np.where instead of if
    #     # if epsilon_total_heart < epsilon_plastic_conductor + epsilon_therm_conductor:
    #     #     return epsilon_total_heart # or epsilon_mecha
    #     # else:
    #     #     return DeformationRte.compute_epsilon_both_materials()

    #     epsilon_plastic = DeformationRte.compute_epsilon_plastic(
    #         T_mean, E, S, polynomial_conductor, max_stress
    #     )
    #     return epsilon_plastic + sigma / E

    # @staticmethod
    # def compute_epsilon_plastic(
    #     T_mean: np.ndarray,
    #     E: np.float64,
    #     S: np.float64,
    #     polynomial: Poly,
    #     max_stress: np.ndarray | None = None,
    # ) -> np.ndarray:
    #     """Computes elastic permanent strain."""
    #     sigma = T_mean / S
    #     if max_stress is None:
    #         max_stress = np.full(T_mean.shape, 0)
    #     # epsilon plastic is based on the highest value between sigma and max_stress
    #     highest_constraint = np.fmax(sigma, max_stress)
    #     equation_solution = DeformationRte.resolve_stress_strain_equation(
    #         highest_constraint, polynomial
    #     )
    #     equation_solution -= highest_constraint / E
    #     return equation_solution

    # @staticmethod
    # def resolve_stress_strain_equation(
    #     sigma: np.ndarray, polynomial: Poly
    # ) -> np.ndarray:
    #     """Solves $\\sigma = Polynomial(\\varepsilon)$"""
    #     polynom_array = np.full(sigma.shape, polynomial)
    #     poly_to_resolve = polynom_array - sigma
    #     return DeformationRte.find_smallest_real_positive_root(poly_to_resolve)

    # @staticmethod
    # def find_smallest_real_positive_root(
    #     poly_to_resolve: np.ndarray,
    # ) -> np.ndarray:
    #     """Find the smallest root that is real and positive for each polynomial

    #     Args:
    #         poly_to_resolve (np.ndarray): array of polynomials to solve

    #     Raises:
    #         ValueError: if no real positive root has been found for at least one polynomial.

    #     Returns:
    #         np.ndarray: array of the roots (one per polynomial)
    #     """
    #     # Can cause performance issues
    #     all_roots = [poly.roots() for poly in poly_to_resolve]

    #     all_roots_stacked = np.stack(all_roots)
    #     keep_solution_condition = np.logical_and(
    #         abs(all_roots_stacked.imag) < IMAGINARY_THRESHOLD,
    #         0.0 <= all_roots_stacked,
    #     )
    #     # Replace roots that are not real nor positive by np.inf
    #     real_positive_roots = np.where(
    #         keep_solution_condition, all_roots_stacked, np.inf
    #     )
    #     real_smallest_root = real_positive_roots.min(axis=1).real
    #     if np.inf in real_smallest_root:
    #         raise ValueError("No solution found for at least one span")
    #     return real_smallest_root

    # @staticmethod
    # def compute_epsilon_therm(
    #     theta: np.ndarray, theta_ref: np.float64, alpha: np.float64
    # ) -> np.ndarray:
    #     """Computing thermal strain using a static method"""
    #     return (theta - theta_ref) * alpha
