# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations

import logging
from typing import Type

import numpy as np
import pandas as pd

import mechaphlowers.core.models.balance.functions as f
from mechaphlowers.core.models.balance import numeric
from mechaphlowers.core.models.balance.find_parameter_solver import (
    FindParamModel,
    FindParamSolver,
    FindParamSolverForLoop,
)
from mechaphlowers.core.models.balance.model_interface import ModelForSolver
from mechaphlowers.core.models.balance.solver import Solver
from mechaphlowers.core.models.balance.utils_balance import (
    Masks,
    VectorProjection,
    fill_to_support,
    reduce_to_span,
)
from mechaphlowers.core.models.cable.deformation import (
    DeformationRte,
    IDeformation,
)
from mechaphlowers.core.models.cable.span import CatenarySpan, Span
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import CableArray, SectionArray

logger = logging.getLogger(__name__)


def nodes_builder(section_array: SectionArray):
    L_chain = section_array.data.insulator_length.to_numpy()
    weight_chain = section_array.data.insulator_weight.to_numpy()
    arm_length = section_array.data.crossarm_length.to_numpy()
    # Convert degrees to rad
    line_angle = f.deg_to_rad(section_array.data.line_angle.to_numpy())
    z = section_array.data.conductor_attachment_altitude.to_numpy()
    span_length = reduce_to_span(section_array.data.span_length.to_numpy())
    load = reduce_to_span(section_array.data.load_weight.to_numpy())
    load_position = reduce_to_span(section_array.data.load_position.to_numpy())
    return Nodes(
        L_chain,
        weight_chain,
        arm_length,
        line_angle,
        z,
        span_length,
        load,
        load_position,
    )


def span_model_builder(
    section_array: SectionArray,
    cable_array: CableArray,
    span_model_type: Type[Span],
):
    span_length = section_array.data.span_length.to_numpy()
    elevation_difference = section_array.data.elevation_difference.to_numpy()
    sagging_parameter = section_array.data.sagging_parameter.to_numpy()
    linear_weight = np.float64(cable_array.data.linear_weight.iloc[0])
    return span_model_type(
        span_length,
        elevation_difference,
        sagging_parameter,
        linear_weight=linear_weight,
    )


def deformation_model_builder(
    cable_array: CableArray,
    span_model: Span,
    sagging_temperature: np.ndarray,
    deformation_model_type: Type[IDeformation] = DeformationRte,
):
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


class BalanceEngine:
    def __init__(
        self,
        cable_array: CableArray,
        section_array: SectionArray,
        span_model_type: Type[Span] = CatenarySpan,
        deformation_model_type: Type[IDeformation] = DeformationRte,
    ):
        self.nodes = nodes_builder(section_array)

        # TODO: fix this
        sagging_temperature = reduce_to_span(
            (section_array.data.sagging_temperature.to_numpy())
        )
        parameter = reduce_to_span(
            section_array.data.sagging_parameter.to_numpy()
        )
        self.span_model = span_model_builder(
            section_array, cable_array, span_model_type
        )
        self.cable_loads = CableLoads(
            np.float64(cable_array.data.diameter.iloc[0]),
            np.float64(cable_array.data.linear_weight.iloc[0]),
            np.zeros_like(self.nodes.weight_chain),
            np.zeros_like(self.nodes.weight_chain),
        )
        self.deformation_model = deformation_model_builder(
            cable_array,
            self.span_model,
            sagging_temperature,
            deformation_model_type,
        )

        self.balance_model = BalanceModel(
            sagging_temperature,
            self.nodes,
            parameter,
            cable_array,
            self.span_model,
            self.deformation_model,
            self.cable_loads,
        )

    def solve_adjustment(self):
        self.balance_model.adjustment = True

        sb = Solver()
        sb.solve(self.balance_model)
        # TODO: fix inside of balanceModel instead (use setters)
        self.span_model.compute_and_store_values()
        self.L_ref = self.balance_model.update_L_ref()

    def solve_change_state(
        self,
        wind_pressure: np.ndarray | None = None,
        ice_thickness: np.ndarray | None = None,
        new_temperature: np.ndarray | None = None,
    ):
        if wind_pressure is not None:
            self.balance_model.cable_loads.wind_pressure = wind_pressure
        if ice_thickness is not None:
            self.balance_model.cable_loads.ice_thickness = ice_thickness
        if new_temperature is not None:
            self.balance_model.sagging_temperature = new_temperature
            self.deformation_model.current_temperature = fill_to_support(
                new_temperature
            )
        self.balance_model.adjustment = False
        self.balance_model.nodes.has_load = True
        self.span_model.load_coefficient = (
            self.balance_model.cable_loads.load_coefficient
        )
        sb = Solver()
        sb.solve(self.balance_model)


class BalanceModel(ModelForSolver):
    def __init__(
        self,
        sagging_temperature: np.ndarray,
        nodes: Nodes,
        parameter: np.ndarray,
        cable_array: CableArray,
        span_model: Span,
        deformation_model: IDeformation,
        cable_loads: CableLoads,
        find_param_solver_type: Type[FindParamSolver] = FindParamSolverForLoop,
    ):
        # tempertaure and parameter size n-1 here
        self.sagging_temperature = sagging_temperature
        self.nodes = nodes
        self.parameter = parameter
        self.cable_array = cable_array
        # TODO: temporary solution to manage cable data, must find a better way to do this
        self.cable_section = np.float64(self.cable_array.data.section.iloc[0])
        self.diameter = np.float64(self.cable_array.data.diameter.iloc[0])
        self.linear_weight = np.float64(
            self.cable_array.data.linear_weight.iloc[0]
        )
        self.young_modulus = np.float64(
            self.cable_array.data.young_modulus.iloc[0]
        )
        self.dilatation_coefficient = np.float64(
            self.cable_array.data.dilatation_coefficient.iloc[0]
        )
        self.temperature_reference = np.float64(
            self.cable_array.data.temperature_reference.iloc[0]
        )
        self.span_model = span_model
        self.deformation_model = deformation_model
        self.cable_loads = cable_loads

        self.adjustment: bool = True
        # TODO: during adjustment computation, perhaps set cable_temperature = 0
        # temperature here is tuning temperature / only in the real span part
        # there is another temperature : change state
        self.nodes.has_load = False

        self.load_model = LoadModel(
            self.cable_array,
            self.nodes.load[self.nodes.load_on_span],
            self.nodes.load_position[self.nodes.load_on_span],
            reduce_to_span(self.span_model.span_length)[
                self.nodes.load_on_span
            ],
            reduce_to_span(self.span_model.elevation_difference)[
                self.nodes.load_on_span
            ],
            self.k_load[self.nodes.load_on_span],
            self.sagging_temperature[self.nodes.load_on_span],
            self.parameter[self.nodes.load_on_span],
        )
        self.find_param_model = FindParamModel(
            self.span_model, self.deformation_model
        )
        self.find_param_solver = find_param_solver_type(self.find_param_model)
        self.update()
        self.nodes.compute_dx_dy_dz()
        self.nodes.vector_projection.set_tensions(
            self.Th, self.Tv_d, self.Tv_g
        )
        self.nodes.compute_moment()

    @property
    def k_load(self):
        # TODO: fix length array issues with mechaphlowers
        """Remove last value for consistency between this and mechaphlowers
        mechaphlowers: one value per support
        here: one value per span (nb_support - 1)
        """
        return self.cable_loads.load_coefficient[:-1]

    @property
    def alpha(self):
        beta = self.beta
        return (
            -np.acos(
                self.a / (self.a**2 + self.b**2 * np.sin(beta) ** 2) ** 0.5
            )
            * np.sign(beta)
            * np.sign(self.b)
        )

    @property
    def beta(self):
        """Remove last value for consistency between this and mechaphlowers
        mechaphlowers: beta = [beta0, beta1, beta2, nan]
        Here: beta = [beta0, beta1, beta2] because the last value refers to the last support (no span related to this support)
        """
        # TODO: check why sign different from what already exists in CableLoads
        return -reduce_to_span(self.cable_loads.load_angle)

    def update_L_ref(self):
        self.deformation_model.tension_mean = self.span_model.T_mean()
        self.deformation_model.cable_length = self.span_model.L
        self.deformation_model.current_temperature = fill_to_support(
            self.sagging_temperature
        )

        # L_ref = reduce_to_span(self.deformation_model.L_ref())
        L_0 = reduce_to_span(self.deformation_model.L_0())
        self.L_ref = L_0
        return L_0

    @property
    def a_prime(self):
        return (self.a**2 + self.b**2 * np.sin(self.beta) ** 2) ** 0.5

    @property
    def b_prime(self):
        return self.b * np.cos(self.beta)

    def compute_Th_and_extremum(self):
        """In this method, a, b, and L_ref may refer to a semi span"""
        cardan_parameter = self.approx_parameter(
            self.a_prime, self.b_prime, self.L_ref, self.sagging_temperature
        )
        cardan_parameter = fill_to_support(cardan_parameter)
        self.find_param_model.set_attributes(
            initial_parameter=cardan_parameter,
            L_ref=fill_to_support(self.L_ref),
        )

        parameter = self.find_param_solver.find_parameter()

        self.span_model.set_parameter(parameter)
        Th = reduce_to_span(self.span_model.T_h())
        x_m = reduce_to_span(self.span_model.x_m)
        x_n = reduce_to_span(self.span_model.x_n)
        return Th, x_m, x_n, reduce_to_span(parameter)

    def update_tensions(self):
        """Compute values of Th, Tv_d and Tv_g.
        There are three cases:
        - adjustment: simple computation
        - change state without loads: run SagTensionSolver once
        - change state with loads: solve LoadModel
        """
        # Case: adjustment
        if self.adjustment:
            Th = reduce_to_span(self.span_model.T_h())
            x_m = reduce_to_span(self.span_model.compute_x_m())
            x_n = reduce_to_span(self.span_model.compute_x_n())

        # Case: change state + no load:
        else:
            Th, x_m, x_n, parameter = self.compute_Th_and_extremum()

            # Case: change state + with load
            if self.nodes.has_load and self.nodes.load_on_span.any():
                self.x_i = self.nodes.load_position * self.a_prime
                # TODO: Don't know why adding x_m is needed
                z_i_m = self.span_model.z_one_point(
                    fill_to_support(self.x_i + x_m)
                )
                z_m = self.span_model.z_one_point(fill_to_support(x_m))
                self.z_i = reduce_to_span(z_i_m - z_m)

                self.solve_xi_zi_loads()

                # Left Th and right Th are equal because balance just got solved (same for parameter)
                parameter[self.nodes.load_on_span] = (
                    self.load_model.span_model_left.sagging_parameter
                )
                Th[self.nodes.load_on_span] = (
                    self.load_model.span_model_left.T_h()
                )
                x_m[self.nodes.load_on_span] = (
                    self.load_model.span_model_left.x_m
                )
                x_n[self.nodes.load_on_span] = (
                    self.load_model.span_model_right.x_n
                )

            self.parameter = parameter
            self.span_model.set_parameter(fill_to_support(parameter))

        self.Tv_g = reduce_to_span(self.span_model.T_v(fill_to_support(x_m)))
        self.Tv_d = -reduce_to_span(self.span_model.T_v(fill_to_support(x_n)))
        self.Th = Th
        self.update_projections()

        return Th

    def solve_xi_zi_loads(self):
        self.load_model.set_all(
            self.L_ref[self.nodes.load_on_span],
            self.x_i[self.nodes.load_on_span],
            self.z_i[self.nodes.load_on_span],
            self.a_prime[self.nodes.load_on_span],
            self.b_prime[self.nodes.load_on_span],
            self.k_load[self.nodes.load_on_span],
            self.sagging_temperature[self.nodes.load_on_span],
            self.parameter[self.nodes.load_on_span],
        )

        solver = Solver()
        solver.solve(
            self.load_model,
            perturb=0.001,
            relax_ratio=0.5,
            stop_condition=1.0,
            relax_power=3,
        )

    def update_projections(self):
        alpha = self.alpha
        beta = self.beta

        self.compute_inter()
        self.nodes.vector_projection.set_all(
            self.Th,
            self.Tv_d,
            self.Tv_g,
            alpha,
            beta,
            self.nodes.line_angle,
            self.proj_angle,
            self.nodes.weight_chain,
        )

    def compute_inter(self):
        # warning: counting from right to left
        proj_d_i = self.nodes.proj_d
        proj_g_ip1 = np.roll(self.nodes.proj_g, -1, axis=1)
        proj_diff = (proj_g_ip1 - proj_d_i)[:, :-1]
        # x: initial input to calculate span_length
        self.inter1 = self.nodes.span_length + proj_diff[0]
        self.inter2 = proj_diff[1]
        self.proj_angle = np.atan2(self.inter2, self.inter1)

    def approx_parameter(self, a, b, L0, cable_temperature):
        circle_chord = (a**2 + b**2) ** 0.5

        factor = (
            self.linear_weight
            * self.k_load
            / self.young_modulus
            / self.cable_section
        )

        p3 = factor * L0
        p2 = (
            L0
            - circle_chord
            + self.dilatation_coefficient * cable_temperature * L0
        )
        p1 = 0 * L0
        p0 = -(a**4) / 24 / circle_chord

        # we have to do p3 * x**3 + p2 * x**2 + p1 * x + p0 = 0
        # p = p3 | p2 | p1 | p0
        p = np.vstack((p3, p2, p1, p0)).T
        roots = numeric.cubic_roots(p)
        return roots.real

    def update_span(self):
        """transmet_portee"""
        # warning for dev : we dont use the first element of span vectors for the moment
        z = self.nodes.z

        self.compute_inter()

        self.a = (self.inter1**2 + self.inter2**2) ** 0.5
        b = np.roll(z, -1) - z
        self.b = b[:-1]
        # TODO: fix array lengths?
        self.span_model.set_lengths(
            fill_to_support(self.a_prime), fill_to_support(self.b_prime)
        )

    def objective_function(self):
        """[Mx0, My0, Mx1, My1,...]"""
        # Need to run update() before this method if necessary
        self.nodes.compute_moment()

        out = np.array([self.nodes.Mx, self.nodes.My]).flatten('F')
        return out

    @property
    def state_vector(self):
        # [dz_0, dy_0, dx_1, dy_1, ... , dz_n, dy_n]
        return self.nodes.state_vector

    @state_vector.setter
    def state_vector(self, value):
        self.nodes.state_vector = value

    def update(self):
        # for the init only, update_span() needs to be call before update_tensions()
        # for other cases, order does not seem to matter
        self.update_span()
        self.update_tensions()

    def dict_to_store(self):
        return {
            "dx": self.nodes.dx,
            "dy": self.nodes.dy,
            "dz": self.nodes.dz,
        }

    def __repr__(self):
        data = {
            'parameter': self.parameter,
            'cable_temperature': self.sagging_temperature,
            'Th': self.Th,
            'Tv_d': self.Tv_d,
            'Tv_g': self.Tv_g,
        }
        out = pd.DataFrame(data)

        return str(out)

    def __str__(self):
        return self.__repr__()


def find_parameter_function(
    parameter: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    L_ref: np.ndarray,
    cable_temperature: np.ndarray,  # float?
    k_load: np.ndarray,
    cable_section: np.float64,
    linear_weight: np.float64,
    dilatation_coefficient: np.float64,
    young_modulus: np.float64,
):
    """this is a placeholder of sagtension algorithm"""
    # TODO: link to mph : SagTensionSolver
    param = parameter

    n_iter = 50
    step = 1.0
    for i in range(n_iter):
        x_m = f.x_m(a, b, param)
        x_n = f.x_n(a, b, param)
        lon = f.L(param, x_n, x_m)
        Tm1 = f.T_moy(
            p=param,
            L=lon,
            x_n=f.x_n(a, b, param),
            x_m=f.x_m(a, b, param),
            linear_weight=linear_weight,
            k_load=k_load,
        )

        delta1 = (lon - L_ref) / L_ref - (
            dilatation_coefficient * cable_temperature
            + Tm1 / young_modulus / cable_section
        )

        mem = param
        param = param + step

        x_m = f.x_m(a, b, param)
        x_n = f.x_n(a, b, param)
        lon = f.L(param, x_n, x_m)
        Tm1 = f.T_moy(
            p=param,
            L=lon,
            x_n=f.x_n(a, b, param),
            x_m=f.x_m(a, b, param),
            linear_weight=linear_weight,
            k_load=k_load,
        )

        delta2 = (lon - L_ref) / L_ref - (
            dilatation_coefficient * cable_temperature
            + Tm1 / young_modulus / cable_section
        )

        param = (param - step) - delta1 / (delta2 - delta1)

        if np.linalg.norm(mem - param) < 0.1 * param.size:
            break
        if i == n_iter:
            logger.info("max iter reached")

    return param


class Nodes:
    def __init__(
        self,
        L_chain: np.ndarray,
        weight_chain: np.ndarray,
        arm_length: np.ndarray,
        line_angle: np.ndarray,
        z: np.ndarray,
        span_length: np.ndarray,
        # load and load_position must of length nb_supports - 1
        load: np.ndarray,
        load_position: np.ndarray,
    ):
        self.L_chain = L_chain
        self.weight_chain = -weight_chain
        # arm length: positive length means further from observer
        self.arm_length = arm_length
        # line_angle: anti clockwise
        self.line_angle = line_angle
        self.init_coordinates(span_length, z)
        # dx, dy, dz are the distances between the attachment point and the arm, including the chain
        self.dxdydz = np.zeros((3, len(z)), dtype=np.float64)
        self.load = load
        self.load_position = load_position
        self.load_on_span = np.logical_and(load != 0, load_position != 0)

        self.has_load = False

        nodes_type = ["suspension"] * len(weight_chain)
        nodes_type[0] = "anchor_first"
        nodes_type[-1] = "anchor_last"
        self.masks = Masks(nodes_type, self.L_chain)

        self.vector_projection = VectorProjection()

    @property
    def dx(self):
        return self.dxdydz[0]

    @dx.setter
    def dx(self, value):
        self.dxdydz[0] = value

    @property
    def dy(self):
        return self.dxdydz[1]

    @dy.setter
    def dy(self, value):
        self.dxdydz[1] = value

    @property
    def dz(self):
        return self.dxdydz[2]

    @dz.setter
    def dz(self, value):
        self.dxdydz[2] = value

    def __len__(self):
        return len(self.L_chain)

    @property
    def z(self):
        """This property returns the altitude of the end the chain.  z = z_arm + dz"""
        return self.z_arm + self.dz

    @property
    def z_arm(self):
        """This property returns the altitude of the end the arm. Should not be modified during computation."""
        return self._z0 - self.z_suspension_chain

    @property
    def proj_g(self):
        arm_length = self.arm_length
        proj_s_axis = -(arm_length + self.dy) * np.sin(self.line_angle / 2) + (
            self.dx
        ) * np.cos(self.line_angle / 2)
        proj_t_axis = (arm_length + self.dy) * np.cos(self.line_angle / 2) + (
            self.dx
        ) * np.sin(self.line_angle / 2)

        return np.array([proj_s_axis, proj_t_axis])

    @property
    def proj_d(self):
        arm_length = self.arm_length
        proj_s_axis = (arm_length + self.dy) * np.sin(self.line_angle / 2) + (
            self.dx
        ) * np.cos(self.line_angle / 2)
        proj_t_axis = (arm_length + self.dy) * np.cos(self.line_angle / 2) - (
            self.dx
        ) * np.sin(self.line_angle / 2)

        return np.array([proj_s_axis, proj_t_axis])

    @property
    def state_vector(self):
        # [dz_0, dy_0, dx_1, dy_1, ... , dz_n, dy_n]
        dxdy = self.dxdydz[[0, 1], 1:-1]
        dzdy = self.dxdydz[[2, 1]][:, [0, -1]]
        return np.vstack([dzdy[:, 0], dxdy.T, dzdy[:, 1]]).flatten()

    @state_vector.setter
    def state_vector(self, state_vector):
        # TODO: refactor with mask
        dzdxdz = state_vector[::2]
        dy = state_vector[1::2]

        self.dy = dy
        self.dx[1:-1] = dzdxdz[1:-1]
        self.dz[[0, -1]] = dzdxdz[[0, -1]]
        self.compute_dx_dy_dz()

    def init_coordinates(self, span_length, z):
        self.x_anchor_chain = np.zeros_like(z)
        self.x_anchor_chain[0] = self.L_chain[0]
        self.x_anchor_chain[-1] = -self.L_chain[-1]
        self.z_suspension_chain = np.zeros_like(z)
        self.z_suspension_chain[1:-1] = -self.L_chain[1:-1]

        # warning: x0 and z0 does not mean the same thing
        # x0 is the absissa of the arm
        # z0 is the altitude of the attachement point
        self.span_length = span_length
        self._z0 = z
        self._y = np.zeros_like(z, dtype=np.float64)

    def compute_dx_dy_dz(self):
        """Update dx and dz according to L_chain and the othre displacement, using Pytagoras
        For anchor chains: update dx
        For suspension chains: update dz
        """
        self.dx, self.dz = self.masks.compute_dx_dy_dz(
            self.dx, self.dy, self.dz
        )

    def compute_moment(self):
        # Placeholder for force computation logic

        Fx, Fy, Fz = self.vector_projection.forces()

        lever_arm = np.array([self.dx, self.dy, self.dz]).T
        # size : (nb nodes , 3 for 3D)

        force_3d = np.vstack((Fx, Fy, Fz)).T

        M = np.cross(lever_arm, force_3d)
        Mx = M[:, 0]
        My = M[:, 1]
        # Mz is supposed to be equal to 0 on each span
        Mz = M[:, 2]
        self.Fx = Fx
        self.Fy = Fy
        self.Fz = Fz
        self.Mx = Mx
        self.My = My
        self.Mz = Mz

    def __repr__(self):
        data = {
            'dx': self.dx,
            'dy': self.dy,
            'dz': self.dz,
            'Fx': self.Fx,
            'Fy': self.Fy,
            'Fz': self.Fz,
            'Mx': self.Mx,
            'My': self.My,
        }
        out = pd.DataFrame(data)

        return str(out)

    def __str__(self):
        return self.__repr__()


class LoadModel(ModelForSolver):
    def __init__(
        self,
        cable_array: CableArray,
        load: np.ndarray,
        load_position: np.ndarray,
        # placeholder values for initialization
        a: np.ndarray,
        b: np.ndarray,
        k_load: np.ndarray,
        temperature: np.ndarray,
        parameter: np.ndarray,
        find_param_solver_type: Type[FindParamSolver] = FindParamSolverForLoop,
    ):
        self.cable_array = cable_array
        self.cable_section = np.float64(self.cable_array.data.section.iloc[0])
        self.diameter = np.float64(self.cable_array.data.diameter.iloc[0])
        self.linear_weight = np.float64(
            self.cable_array.data.linear_weight.iloc[0]
        )
        self.young_modulus = np.float64(
            self.cable_array.data.young_modulus.iloc[0]
        )
        self.dilatation_coefficient = np.float64(
            self.cable_array.data.dilatation_coefficient.iloc[0]
        )
        self.load = load
        self.load_position = load_position
        self.find_param_solver_type = find_param_solver_type
        # Need a way to choose Span model

        # init objects with placeholder values, but they will be updated when needed
        self.span_model_left = CatenarySpan(
            a, b, parameter, k_load, self.linear_weight
        )
        self.span_model_right = CatenarySpan(
            a, b, parameter, k_load, self.linear_weight
        )
        self.deformation_model_left = deformation_model_builder(
            self.cable_array,
            self.span_model_left,
            temperature,
        )
        self.deformation_model_right = deformation_model_builder(
            self.cable_array,
            self.span_model_right,
            temperature,
        )
        self.find_param_model_left = FindParamModel(
            self.span_model_left, self.deformation_model_left
        )
        self.find_param_model_right = FindParamModel(
            self.span_model_right, self.deformation_model_right
        )
        self.find_param_solver_left = self.find_param_solver_type(
            self.find_param_model_left
        )
        self.find_param_solver_right = self.find_param_solver_type(
            self.find_param_model_right
        )

    def set_all(
        self,
        L_ref: np.ndarray,
        x_i: np.ndarray,
        z_i: np.ndarray,
        a_prime: np.ndarray,
        b_prime: np.ndarray,
        k_load: np.ndarray,
        temperature,
        # placeholder value
        parameter,
    ):
        # put L_ref in constructor?
        self.L_ref = L_ref
        self.x_i = x_i
        self.z_i = z_i
        self.a_prime = a_prime
        self.b_prime = b_prime
        self.k_load = k_load
        self.temperature = temperature
        # Warning: arrays legnth are n-1
        # Easier to use, but less consistent with other mechaphlowers objects
        self.update_objects()

    def update_objects(self):
        self.update_lengths_span_models()
        self.span_model_left.load_coefficient = self.k_load
        self.span_model_right.load_coefficient = self.k_load

        self.deformation_model_left.current_temperature = self.temperature
        self.deformation_model_right.current_temperature = self.temperature

    def update_lengths_span_models(self):
        self.span_model_left.span_length = self.x_i
        self.span_model_left.elevation_difference = self.z_i
        self.span_model_right.span_length = self.a_prime - self.x_i
        self.span_model_right.elevation_difference = self.b_prime - self.z_i

    @property
    def state_vector(self):
        return np.array([self.x_i, self.z_i]).flatten('F')

    @state_vector.setter
    def state_vector(self, value):
        self.x_i = value[::2]
        self.z_i = value[1::2]
        self.update_lengths_span_models()

    def objective_function(self):
        """[Th_diff0, Tv_diff0, Th_diff1, Tv_diff1,...]"""
        # left side of the load
        Th_left = self.span_model_left.T_h()
        # need to update?
        x_n_left = self.span_model_left.x_n

        Tv_d_loc = -self.span_model_left.T_v(x_n_left)

        # right
        Th_right = self.span_model_right.T_h()
        x_m_right = self.span_model_right.x_m

        Tv_g_loc = self.span_model_right.T_v(x_m_right)

        Th_diff = Th_left - Th_right
        Tv_diff = Tv_d_loc + Tv_g_loc - self.load * self.k_load

        return np.array([Th_diff, Tv_diff]).flatten('F')

    def approx_parameter(self, a, b, L0, cable_temperature):
        # extract this method? There is a duplicate in Span
        circle_chord = (a**2 + b**2) ** 0.5

        factor = (
            self.linear_weight
            * self.k_load
            / self.young_modulus
            / self.cable_section
        )

        p3 = factor * L0
        p2 = (
            L0
            - circle_chord
            + self.dilatation_coefficient * cable_temperature * L0
        )
        p1 = 0 * L0
        p0 = -(a**4) / 24 / circle_chord

        # we have to do p3 * x**3 + p2 * x**2 + p1 * x + p0 = 0
        # p = p3 | p2 | p1 | p0
        p = np.vstack((p3, p2, p1, p0)).T
        roots = numeric.cubic_roots(p)
        return roots.real

    def update(self):
        self.update_lengths_span_models()
        # update parameter left
        L_ref_left = self.load_position * self.L_ref
        a_left = self.x_i
        b_left = self.z_i
        parameter_left = self.approx_parameter(
            a_left, b_left, L_ref_left, self.temperature
        )
        self.find_param_model_left.set_attributes(
            initial_parameter=parameter_left,
            L_ref=L_ref_left,
        )

        parameter_left = self.find_param_solver_left.find_parameter()
        # parameter_left = find_parameter_function(
        #     parameter_left,
        #     a_left,
        #     b_left,
        #     L_ref_left,
        #     self.temperature,
        #     self.k_load,
        #     self.cable_section,
        #     self.linear_weight,
        #     self.dilatation_coefficient,
        #     self.young_modulus,
        # )
        self.span_model_left.set_parameter(parameter_left)

        # update parameter right
        L_ref_right = self.L_ref - L_ref_left
        a_right = self.a_prime - self.x_i
        b_right = self.b_prime - self.z_i
        parameter_right = self.approx_parameter(
            a_right, b_right, L_ref_right, self.temperature
        )
        self.find_param_model_right.set_attributes(
            initial_parameter=parameter_right,
            L_ref=L_ref_right,
        )

        parameter_right = self.find_param_solver_right.find_parameter()
        # parameter_right = find_parameter_function(
        #     parameter_right,
        #     a_right,
        #     b_right,
        #     L_ref_right,
        #     self.temperature,
        #     self.k_load,
        #     self.cable_section,
        #     self.linear_weight,
        #     self.dilatation_coefficient,
        #     self.young_modulus,
        # )
        self.span_model_right.set_parameter(parameter_right)
