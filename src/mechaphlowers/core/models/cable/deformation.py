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
    # This class may be irrelevant, these methods and attributes could be implemented directly into DeformationRte
    def __init__(
        self,
        stress_strain_polynomial: Poly,
        young_modulus: np.float64,
        dilatation_coefficient: np.float64,
        T_labo: np.float64,
    ) -> None:
        self.stress_strain_polynomial = stress_strain_polynomial
        self.young_modulus = young_modulus
        self.dilatation_coefficient = dilatation_coefficient
        self.T_labo = T_labo  # rename temp ref?
        self.sigma_max: np.ndarray | None = None
        self.epsilon_plastic_max: np.ndarray | None = None

    def epsilon_th(self, current_temperature: np.ndarray) -> np.ndarray:
        """Computes the thermal strain of the material.
        Args:
            current_temperature (np.ndarray): current temperature
        Returns:
            np.ndarray: thermal strain
        """
        return self.dilatation_coefficient * (
            current_temperature - self.T_labo
        )


class IDeformation(ABC):
    """This abstract class is a base class for models to compute relative cable deformations."""

    def __init__(
        self,
        data_cable: DataCable,
        tension_mean: np.ndarray,
        cable_length: np.ndarray,
        sagging_temperature: np.ndarray,
        max_stress: np.ndarray | None = None,
        **kwargs,
    ):
        self.tension_mean = tension_mean
        self.cable_length = cable_length
        self.data_cable = data_cable
        self.current_temperature = sagging_temperature

    # Keep this in abstract class?
    @property
    @abstractmethod
    def max_stress(self):
        pass

    @max_stress.setter
    @abstractmethod
    def max_stress(self, value_max_stress):
        pass

    @abstractmethod
    def L_ref(self) -> np.ndarray:
        """Unstressed cable length, at a chosen reference temperature"""

    @abstractmethod
    def epsilon(self) -> np.ndarray:
        """Total relative strain of the cable."""

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


class DeformationOneMaterial(IDeformation):
    """This class implements the deformation model used by RTE."""
    def __init__(self, polynomial:Poly, young_modulus, dilatation_coefficient: float, T_labo: float, **kwargs):
        self.poly = polynomial
        self.young_modulus = young_modulus
        self.dilatation_coefficient = dilatation_coefficient
        self.T_labo = T_labo  # rename temp ref?
        # self.set_epsilon_max impossible: we did not know size
        
    def set_epsilon_max(self,epsilon_max):
        self.epsilon_max = epsilon_max
        self.sigma_max = self.poly(epsilon_max)
        self.sigma_min = self.sigma_max - self.young_modulus * epsilon_max
        self.epsilon_0 = epsilon_max - self.sigma_max / self.young_modulus
        
    def plastic_func(self):
        """Compute the plastic strain of a material given its stress"""
        return lambda epsilon: self.young_modulus * epsilon + self.sigma_min
    
    def inverse_plastic_func(self):
        """Compute the plastic strain of a material given its stress"""
        return lambda sigma: (sigma - self.sigma_min) / self.young_modulus
        
    def get_strain(self, stress):
        """Compute the strain of a material given its stress"""
        strain = self.inverse_plastic_func()(stress)
        return np.where(strain < 0, 0.0, strain)
    
    def epsilon_th(self, current_temperature: np.ndarray) -> np.ndarray:
        """Computes the thermal strain of the material.
        Args:
            current_temperature (np.ndarray): current temperature
        Returns:
            np.ndarray: thermal strain
        """
        return self.dilatation_coefficient * (
            current_temperature - self.T_labo
        )
        
    def build_array_polynomials(
        self, current_temperature: np.ndarray
    ) -> np.ndarray:
        """Creates an array of polynomials. Each polynomial is translated along the x axis by the value of epsilon_th."""
        eps_th = self.epsilon_th(current_temperature)
        polynomials_array = np.full(current_temperature.shape, self.poly)
        polynomials_translated = [
            polynomials_array[index].convert(
                domain=[eps_th[index] - 1, eps_th[index] + 1]
            )
            for index in range(len(polynomials_array))
        ]
        return np.array(polynomials_translated)
        
class DeformationCable:
    def __init__(self, cable_section_area, 
                 polynomial_conductor:Poly, young_modulus_conductor, dilatation_coefficient_conductor: float,
                 polynomial_heart:Poly, young_modulus_heart, dilatation_coefficient_heart: float,T_labo: float, **kwargs):
        # self.poly = polynomial
        # self.young_modulus = young_modulus
        # self.dilatation_coefficient = dilatation_coefficient
        # self.T_labo = T_labo    
        self.cable_section = cable_section_area
        self.conductor_layer = DeformationOneMaterial(polynomial_conductor, young_modulus_conductor, dilatation_coefficient_conductor, T_labo)
        self.monolayer_cable = True
        # self.operating_regime = 
        self.heart_layer = DeformationOneMaterial(polynomial_heart, young_modulus_heart, dilatation_coefficient_heart, T_labo)
        if young_modulus_heart > np.float(0.0):
            self.monolayer_cable = False
        
    @property    
    def sigma_1(self, eps_total):
        conductor_epsilon_0 = self.conductor_layer.epsilon_0
        return self.heart_layer.sigma_0(conductor_epsilon_0)
    
    @property
    def sigma_max(self):
        return self._sigma_max
    
    @sigma_max.setter
    def sigma_max(self, value):
        self._sigma_max = value
        epsilon_max = self.solve_epsilon_above_max_stress(value)
        self.conductor_layer.epsilon_max = epsilon_max
        

    
    def inverse_plastic_func(self, current_temperature, operating_mode):
        """Compute the plastic strain of a material given its stress"""
        eps_th_heart = self.heart_layer.epsilon_th(current_temperature)
        eps_th_conductor = self.conductor_layer.epsilon_th(current_temperature)
        young_modulus = self.conductor_layer.young_modulus
        sigma_min = self.conductor_layer.sigma_min
        thermic_shift = young_modulus * eps_th_conductor
        
        # if operating_mode == "affine":
        #     young_modulus = self.conductor_layer.young_modulus 
        #     sigma_min = self.conductor_layer.sigma_min 
        if operating_mode == "sum":
            young_modulus = self.conductor_layer.young_modulus + self.heart_layer.young_modulus
            sigma_min = self.conductor_layer.sigma_min + self.heart_layer.sigma_min
            thermic_shift = self.conductor_layer.young_modulus * eps_th_conductor + self.heart_layer.young_modulus * eps_th_heart

        return lambda sigma: (sigma - sigma_min) + thermic_shift / young_modulus
        
    def get_strain(self, tension_mean, current_temperature):
        """Compute the strain of a material given its stress"""
        stress = tension_mean / self.cable_section
        strain = np.where(stress < 0, 0.0, 0.0)
        strain = np.where(stress < self.sigma_1 & stress >= 0, strain, self.inverse_plastic_func()(stress))
        strain = np.where(stress < self.sigma_max & stress >= self.sigma_1, strain, self.inverse_plastic_func(sum)(stress))
        # np.extract plutot
        need_solving = np.logical_and(
                stress > self.sigma_max, np.logical_not(np.isnan(stress))
            )
        sigma_solving = np.extract(need_solving, stress)
        if len(sigma_solving) > 0:
            eps_tot_solved = self.solve_epsilon_above_max_stress(
                sigma_solving,
                current_temperature,
            )
            np.place(
                strain,
                need_solving,
                eps_tot_solved,
            )     
        return strain

    
    def solve_epsilon_above_max_stress(
        self, sigma: np.ndarray, current_temperature: np.ndarray,
    ) -> np.ndarray:
        # create arrays of polynomials where eps_th are taken into account
        poly_array_heart = self.heart_layer.build_array_polynomials(current_temperature)
        poly_array_conductor = self.conductor_layer.build_array_polynomials(current_temperature)

        # stress value where cable is only supported by heart material if below this value
        eps_translated_conductor = (
            self.conductor_layer.epsilon_plastic_max
            + self.conductor_layer.epsilon_th(current_temperature)
        )
        sigma_switch_point = []
        for index in range(len(eps_translated_conductor)):
            eps = eps_translated_conductor[index]
            sigma_switch_point.append(poly_array_heart[index](eps))

        is_only_heart_supported = sigma < sigma_switch_point

        polynomials_to_resolve = np.where(
            is_only_heart_supported,
            poly_array_heart - sigma,  # case only heart
            poly_array_heart
            + poly_array_conductor
            - sigma,  # case both materials
        )
        eps_tot = np.array(
            [
                self.find_smallest_real_positive_root(
                    single_poly_to_resolve
                )
                for single_poly_to_resolve in polynomials_to_resolve
            ]
        )
        return np.array(eps_tot)



    @staticmethod
    def find_smallest_real_positive_root(
        polynomial_to_resolve: Poly,
    ) -> np.float64:
        """Find the smallest root that is real and positive for one polynomial

        Args:
                polynomial_to_resolve (np.ndarray): single polynomial to solve

        Raises:
                ValueError: if no real positive root has been found.

        Returns:
                np.float64: smallest root (real and positive)
        """
        all_roots: np.ndarray = polynomial_to_resolve.roots()
        keep_solution_condition = np.logical_and(
            abs(all_roots.imag) < IMAGINARY_THRESHOLD,
            0.0 <= all_roots,
        )
        real_positive_roots = all_roots[keep_solution_condition]
        if real_positive_roots.size == 0:
            raise ValueError("No solution found for at least one span")
        return real_positive_roots.real.min()



class DeformationRteRefactor(IDeformation):
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
        self.sigma_func_conductor = SigmaFunctionSingleMaterial(
            **data_cable.conductor_material_dict
        )
        self.sigma_func_heart = SigmaFunctionSingleMaterial(
            **data_cable.heart_material_dict
        )

        if max_stress is None:
            self._max_stress = np.zeros_like(self.cable_length)
            self.sigma_func_conductor.sigma_max = max_stress
            self.sigma_func_conductor.epsilon_plastic_max = np.zeros_like(
                tension_mean
            )
            self.sigma_func_heart.sigma_max = max_stress
            self.sigma_func_heart.epsilon_plastic_max = np.zeros_like(
                tension_mean
            )
        else:
            # TODO: case not implemented correctly
            self.max_stress = max_stress

    @property
    def max_stress(self) -> np.ndarray:
        return self._max_stress

    @max_stress.setter
    def max_stress(self, value_max_stress: np.ndarray):
        self._max_stress = value_max_stress
        # recompute when setting a new max_stress in order to store a new epsilon_plastic
        eps_tot = DeformationRte.solve_epsilon_above_max_stress(
            value_max_stress,
            self.data_cable,
            self.current_temperature,
            self.sigma_func_conductor,
            self.sigma_func_heart,
        )
        eps_plastic_conductor = (
            eps_tot
            - value_max_stress / self.data_cable.young_modulus_conductor
        )
        self.sigma_func_conductor.epsilon_plastic_max = eps_plastic_conductor
        if self.data_cable.young_modulus_heart != np.float64(0):
            eps_plastic_heart = (
                eps_tot
                - value_max_stress / self.data_cable.young_modulus_heart
            )
            self.sigma_func_heart.epsilon_plastic_max = eps_plastic_heart

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
        if max_stress is None:
            max_stress = np.zeros_like(T_mean)
            # not pretty but didn't find better
            # currently, SagTensionSolver cannot update max_stress because of the use of static methods
            sigma_func_conductor.sigma_max = max_stress
            sigma_func_conductor.epsilon_plastic_max = np.zeros_like(T_mean)
            sigma_func_heart.sigma_max = max_stress
            sigma_func_heart.epsilon_plastic_max = np.zeros_like(T_mean)
        S = data_cable.cable_section_area
        sigma = T_mean / S
        # single material case (therefore linear)
        if data_cable.young_modulus_heart == np.float64(0):
            eps_mecha = DeformationRte.formula_epsilon_mecha_one_material(
                T_mean, data_cable.young_modulus, S
            )
            eps_th = sigma_func_conductor.epsilon_th(current_temperature)
            return eps_mecha + eps_th
        # two materials
        else:
            # case where analytical solution is enough
            # compute for all spans even if not necessary
            eps_tot = DeformationRte.compute_epsilon_below_max_stress(
                sigma,
                current_temperature,
                sigma_func_conductor,
                sigma_func_heart,
            )
            # extract spans where equation resolution is needed
            # compute only for spans needed, for performance reasons
            need_solving = np.logical_and(
                sigma > max_stress, np.logical_not(np.isnan(sigma))
            )
            sigma_solving = np.extract(need_solving, sigma)
            if len(sigma_solving) > 0:
                eps_tot_solved = DeformationRte.solve_epsilon_above_max_stress(
                    sigma_solving,
                    data_cable,
                    current_temperature,
                    sigma_func_conductor,
                    sigma_func_heart,
                )
                np.place(
                    eps_tot,
                    need_solving,
                    eps_tot_solved,
                )
            return eps_tot

    @staticmethod
    def formula_epsilon_mecha_one_material(
        T_mean: np.ndarray,
        E: np.float64,
        S: np.float64,
    ) -> np.ndarray:
        return T_mean / (E * S)

    @staticmethod
    def compute_epsilon_below_max_stress(
        sigma: np.ndarray,
        current_temperature: np.ndarray,
        sigma_func_conductor: SigmaFunctionSingleMaterial,
        sigma_func_heart: SigmaFunctionSingleMaterial,
    ) -> np.ndarray:
        # case where cable is only supported by the heart material
        eps_total_heart_only = DeformationRte.formula_eps_linear_heart(
            sigma, current_temperature, sigma_func_heart
        )
        # case where both materials need to be taken into account
        eps_total_both_materials = (
            DeformationRte.formula_eps_linear_both_materials(
                sigma,
                current_temperature,
                sigma_func_heart,
                sigma_func_conductor,
            )
        )

        is_only_heart_supported: np.ndarray = (
            eps_total_heart_only
            < sigma_func_conductor.epsilon_plastic_max
            + sigma_func_conductor.epsilon_th(current_temperature)
        )

        return np.where(
            is_only_heart_supported,
            eps_total_heart_only,
            eps_total_both_materials,
        )

    @staticmethod
    def formula_eps_linear_heart(
        sigma: np.ndarray,
        current_temperature: np.ndarray,
        sigma_func_heart: SigmaFunctionSingleMaterial,
    ) -> np.ndarray:
        """$$\\varepsilon_{total} = \\frac{\\sigma}{E_{heart}}
        + \\varepsilon_{plastic, heart} + \\varepsilon_{th, heart}$$"""
        E = sigma_func_heart.young_modulus
        eps_th = sigma_func_heart.epsilon_th(current_temperature)
        eps_plastic = sigma_func_heart.epsilon_plastic_max
        return sigma / E + eps_th + eps_plastic

    @staticmethod
    def formula_eps_linear_both_materials(
        sigma: np.ndarray,
        current_temperature: np.ndarray,
        sigma_func_heart: SigmaFunctionSingleMaterial,
        sigma_func_conductor: SigmaFunctionSingleMaterial,
    ) -> np.ndarray:
        """$$\\varepsilon_{total} = \\frac{1}{E_{heart} + E_{conductor}} * (\\sigma
        + (\\varepsilon_{plastic, heart} + \\varepsilon_{th, heart}) * E_{heart}
        + (\\varepsilon_{plastic, conductor} + \\varepsilon_{th, conductor}) * E_{conductor})$$"""
        E_a = sigma_func_conductor.young_modulus
        eps_th_a = sigma_func_conductor.epsilon_th(current_temperature)
        eps_plastic_a = sigma_func_conductor.epsilon_plastic_max
        E_b = sigma_func_heart.young_modulus
        eps_th_b = sigma_func_heart.epsilon_th(current_temperature)
        eps_plastic_b = sigma_func_heart.epsilon_plastic_max
        return (
            sigma
            + (eps_th_a + eps_plastic_a) * E_a
            + (eps_th_b + eps_plastic_b) * E_b
        ) / (E_a + E_b)

    @staticmethod
    def solve_epsilon_above_max_stress(
        sigma: np.ndarray,
        data_cable: DataCable,
        current_temperature: np.ndarray,
        sigma_func_conductor: SigmaFunctionSingleMaterial,
        sigma_func_heart: SigmaFunctionSingleMaterial,
    ) -> np.ndarray:
        # create arrays of polynomials where eps_th are taken into account
        poly_array_heart = DeformationRte.build_array_polynomials(
            data_cable.polynomial_heart, sigma_func_heart, current_temperature
        )
        poly_array_conductor = DeformationRte.build_array_polynomials(
            data_cable.polynomial_conductor,
            sigma_func_conductor,
            current_temperature,
        )

        # stress value where cable is only supported by heart material if below this value
        eps_translated_conductor = (
            sigma_func_conductor.epsilon_plastic_max
            + sigma_func_conductor.epsilon_th(current_temperature)
        )
        sigma_switch_point = []
        for index in range(len(eps_translated_conductor)):
            eps = eps_translated_conductor[index]
            sigma_switch_point.append(poly_array_heart[index](eps))

        is_only_heart_supported = sigma < sigma_switch_point

        polynomials_to_resolve = np.where(
            is_only_heart_supported,
            poly_array_heart - sigma,  # case only heart
            poly_array_heart
            + poly_array_conductor
            - sigma,  # case both materials
        )
        eps_tot = np.array(
            [
                DeformationRte.find_smallest_real_positive_root(
                    single_poly_to_resolve
                )
                for single_poly_to_resolve in polynomials_to_resolve
            ]
        )
        return np.array(eps_tot)

    @staticmethod
    def build_array_polynomials(
        polynomial: Poly,
        sigma_func: SigmaFunctionSingleMaterial,
        current_temperature: np.ndarray,
    ) -> np.ndarray:
        """Creates an array of polynomials. Each polynomial is translated along the x axis by the value of epsilon_th."""
        eps_th = sigma_func.epsilon_th(current_temperature)
        polynomials_array = np.full(current_temperature.shape, polynomial)
        polynomials_translated = [
            polynomials_array[index].convert(
                domain=[eps_th[index] - 1, eps_th[index] + 1]
            )
            for index in range(len(polynomials_array))
        ]
        return np.array(polynomials_translated)

    @staticmethod
    def find_smallest_real_positive_root(
        polynomial_to_resolve: Poly,
    ) -> np.float64:
        """Find the smallest root that is real and positive for one polynomial

        Args:
                polynomial_to_resolve (np.ndarray): single polynomial to solve

        Raises:
                ValueError: if no real positive root has been found.

        Returns:
                np.float64: smallest root (real and positive)
        """
        all_roots: np.ndarray = polynomial_to_resolve.roots()
        keep_solution_condition = np.logical_and(
            abs(all_roots.imag) < IMAGINARY_THRESHOLD,
            0.0 <= all_roots,
        )
        real_positive_roots = all_roots[keep_solution_condition]
        if real_positive_roots.size == 0:
            raise ValueError("No solution found for at least one span")
        return real_positive_roots.real.min()


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
        self.sigma_func_conductor = SigmaFunctionSingleMaterial(
            **data_cable.conductor_material_dict
        )
        self.sigma_func_heart = SigmaFunctionSingleMaterial(
            **data_cable.heart_material_dict
        )

        if max_stress is None:
            self._max_stress = np.zeros_like(self.cable_length)
            self.sigma_func_conductor.sigma_max = max_stress
            self.sigma_func_conductor.epsilon_plastic_max = np.zeros_like(
                tension_mean
            )
            self.sigma_func_heart.sigma_max = max_stress
            self.sigma_func_heart.epsilon_plastic_max = np.zeros_like(
                tension_mean
            )
        else:
            # TODO: case not implemented correctly
            self.max_stress = max_stress

    @property
    def max_stress(self) -> np.ndarray:
        return self._max_stress

    @max_stress.setter
    def max_stress(self, value_max_stress: np.ndarray):
        self._max_stress = value_max_stress
        # recompute when setting a new max_stress in order to store a new epsilon_plastic
        eps_tot = DeformationRte.solve_epsilon_above_max_stress(
            value_max_stress,
            self.data_cable,
            self.current_temperature,
            self.sigma_func_conductor,
            self.sigma_func_heart,
        )
        eps_plastic_conductor = (
            eps_tot
            - value_max_stress / self.data_cable.young_modulus_conductor
        )
        self.sigma_func_conductor.epsilon_plastic_max = eps_plastic_conductor
        if self.data_cable.young_modulus_heart != np.float64(0):
            eps_plastic_heart = (
                eps_tot
                - value_max_stress / self.data_cable.young_modulus_heart
            )
            self.sigma_func_heart.epsilon_plastic_max = eps_plastic_heart

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
        if max_stress is None:
            max_stress = np.zeros_like(T_mean)
            # not pretty but didn't find better
            # currently, SagTensionSolver cannot update max_stress because of the use of static methods
            sigma_func_conductor.sigma_max = max_stress
            sigma_func_conductor.epsilon_plastic_max = np.zeros_like(T_mean)
            sigma_func_heart.sigma_max = max_stress
            sigma_func_heart.epsilon_plastic_max = np.zeros_like(T_mean)
        S = data_cable.cable_section_area
        sigma = T_mean / S
        # single material case (therefore linear)
        if data_cable.young_modulus_heart == np.float64(0):
            eps_mecha = DeformationRte.formula_epsilon_mecha_one_material(
                T_mean, data_cable.young_modulus, S
            )
            eps_th = sigma_func_conductor.epsilon_th(current_temperature)
            return eps_mecha + eps_th
        # two materials
        else:
            # case where analytical solution is enough
            # compute for all spans even if not necessary
            eps_tot = DeformationRte.compute_epsilon_below_max_stress(
                sigma,
                current_temperature,
                sigma_func_conductor,
                sigma_func_heart,
            )
            # extract spans where equation resolution is needed
            # compute only for spans needed, for performance reasons
            need_solving = np.logical_and(
                sigma > max_stress, np.logical_not(np.isnan(sigma))
            )
            sigma_solving = np.extract(need_solving, sigma)
            if len(sigma_solving) > 0:
                eps_tot_solved = DeformationRte.solve_epsilon_above_max_stress(
                    sigma_solving,
                    data_cable,
                    current_temperature,
                    sigma_func_conductor,
                    sigma_func_heart,
                )
                np.place(
                    eps_tot,
                    need_solving,
                    eps_tot_solved,
                )
            return eps_tot

    @staticmethod
    def formula_epsilon_mecha_one_material(
        T_mean: np.ndarray,
        E: np.float64,
        S: np.float64,
    ) -> np.ndarray:
        return T_mean / (E * S)

    @staticmethod
    def compute_epsilon_below_max_stress(
        sigma: np.ndarray,
        current_temperature: np.ndarray,
        sigma_func_conductor: SigmaFunctionSingleMaterial,
        sigma_func_heart: SigmaFunctionSingleMaterial,
    ) -> np.ndarray:
        # case where cable is only supported by the heart material
        eps_total_heart_only = DeformationRte.formula_eps_linear_heart(
            sigma, current_temperature, sigma_func_heart
        )
        # case where both materials need to be taken into account
        eps_total_both_materials = (
            DeformationRte.formula_eps_linear_both_materials(
                sigma,
                current_temperature,
                sigma_func_heart,
                sigma_func_conductor,
            )
        )

        is_only_heart_supported: np.ndarray = (
            eps_total_heart_only
            < sigma_func_conductor.epsilon_plastic_max
            + sigma_func_conductor.epsilon_th(current_temperature)
        )

        return np.where(
            is_only_heart_supported,
            eps_total_heart_only,
            eps_total_both_materials,
        )

    @staticmethod
    def formula_eps_linear_heart(
        sigma: np.ndarray,
        current_temperature: np.ndarray,
        sigma_func_heart: SigmaFunctionSingleMaterial,
    ) -> np.ndarray:
        """$$\\varepsilon_{total} = \\frac{\\sigma}{E_{heart}}
        + \\varepsilon_{plastic, heart} + \\varepsilon_{th, heart}$$"""
        E = sigma_func_heart.young_modulus
        eps_th = sigma_func_heart.epsilon_th(current_temperature)
        eps_plastic = sigma_func_heart.epsilon_plastic_max
        return sigma / E + eps_th + eps_plastic

    @staticmethod
    def formula_eps_linear_both_materials(
        sigma: np.ndarray,
        current_temperature: np.ndarray,
        sigma_func_heart: SigmaFunctionSingleMaterial,
        sigma_func_conductor: SigmaFunctionSingleMaterial,
    ) -> np.ndarray:
        """$$\\varepsilon_{total} = \\frac{1}{E_{heart} + E_{conductor}} * (\\sigma
        + (\\varepsilon_{plastic, heart} + \\varepsilon_{th, heart}) * E_{heart}
        + (\\varepsilon_{plastic, conductor} + \\varepsilon_{th, conductor}) * E_{conductor})$$"""
        E_a = sigma_func_conductor.young_modulus
        eps_th_a = sigma_func_conductor.epsilon_th(current_temperature)
        eps_plastic_a = sigma_func_conductor.epsilon_plastic_max
        E_b = sigma_func_heart.young_modulus
        eps_th_b = sigma_func_heart.epsilon_th(current_temperature)
        eps_plastic_b = sigma_func_heart.epsilon_plastic_max
        return (
            
        

    @staticmethod
    def solve_epsilon_above_max_stress(
        sigma: np.ndarray,
        data_cable: DataCable,
        current_temperature: np.ndarray,
        sigma_func_conductor: SigmaFunctionSingleMaterial,
        sigma_func_heart: SigmaFunctionSingleMaterial,
    ) -> np.ndarray:
        # create arrays of polynomials where eps_th are taken into account
        poly_array_heart = DeformationRte.build_array_polynomials(
            data_cable.polynomial_heart, sigma_func_heart, current_temperature
        )
        poly_array_conductor = DeformationRte.build_array_polynomials(
            data_cable.polynomial_conductor,
            sigma_func_conductor,
            current_temperature,
        )

        # stress value where cable is only supported by heart material if below this value
        eps_translated_conductor = (
            sigma_func_conductor.epsilon_plastic_max
            + sigma_func_conductor.epsilon_th(current_temperature)
        )
        sigma_switch_point = []
        for index in range(len(eps_translated_conductor)):
            eps = eps_translated_conductor[index]
            sigma_switch_point.append(poly_array_heart[index](eps))

        is_only_heart_supported = sigma < sigma_switch_point

        polynomials_to_resolve = np.where(
            is_only_heart_supported,
            poly_array_heart - sigma,  # case only heart
            poly_array_heart
            + poly_array_conductor
            - sigma,  # case both materials
        )
        eps_tot = np.array(
            [
                DeformationRte.find_smallest_real_positive_root(
                    single_poly_to_resolve
                )
                for single_poly_to_resolve in polynomials_to_resolve
            ]
        )
        return np.array(eps_tot)

    @staticmethod
    def build_array_polynomials(
        polynomial: Poly,
        sigma_func: SigmaFunctionSingleMaterial,
        current_temperature: np.ndarray,
    ) -> np.ndarray:
        """Creates an array of polynomials. Each polynomial is translated along the x axis by the value of epsilon_th."""
        eps_th = sigma_func.epsilon_th(current_temperature)
        polynomials_array = np.full(current_temperature.shape, polynomial)
        polynomials_translated = [
            polynomials_array[index].convert(
                domain=[eps_th[index] - 1, eps_th[index] + 1]
            )
            for index in range(len(polynomials_array))
        ]
        return np.array(polynomials_translated)

    @staticmethod
    def find_smallest_real_positive_root(
        polynomial_to_resolve: Poly,
    ) -> np.float64:
        """Find the smallest root that is real and positive for one polynomial

        Args:
                polynomial_to_resolve (np.ndarray): single polynomial to solve

        Raises:
                ValueError: if no real positive root has been found.

        Returns:
                np.float64: smallest root (real and positive)
        """
        all_roots: np.ndarray = polynomial_to_resolve.roots()
        keep_solution_condition = np.logical_and(
            abs(all_roots.imag) < IMAGINARY_THRESHOLD,
            0.0 <= all_roots,
        )
        real_positive_roots = all_roots[keep_solution_condition]
        if real_positive_roots.size == 0:
            raise ValueError("No solution found for at least one span")
        return real_positive_roots.real.min()
