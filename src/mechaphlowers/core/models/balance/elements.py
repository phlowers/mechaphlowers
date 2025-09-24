# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Type

import numpy as np
import pandas as pd

import mechaphlowers.core.models.balance.functions as f
from mechaphlowers.core.models.balance import numeric
from mechaphlowers.core.models.balance.model_interface import ModelForSolver
from mechaphlowers.core.models.balance.solver import Solver
from mechaphlowers.core.models.balance.utils_balance import (
    VectorProjection,
    fill_to_support,
    reduce_to_span,
)
from mechaphlowers.core.models.cable.span import CatenarySpan, Span
from mechaphlowers.core.models.external_loads import CableLoads
from mechaphlowers.entities.arrays import CableArray, SectionArray

logger = logging.getLogger(__name__)


@dataclass
class Cable:
    section: np.float64  # input in mm ?
    linear_weight: np.float64
    dilatation_coefficient: np.float64
    young_modulus: np.float64
    diameter: np.float64
    cra: np.float64


def section_array_to_nodes(section_array: SectionArray):
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


def arrays_to_span_model(
    section_array: SectionArray,
    cable_array: CableArray,
    span_model: Type[Span],
):
    span_length = section_array.data.span_length.to_numpy()
    elevation_difference = section_array.data.elevation_difference.to_numpy()
    sagging_parameter = section_array.data.sagging_parameter.to_numpy()
    linear_weight = np.float64(cable_array.data.linear_weight)
    return span_model(
        span_length,
        elevation_difference,
        sagging_parameter,
        linear_weight=linear_weight,
    )


class BalanceEngine:
    def __init__(
        self,
        cable_array: CableArray,
        section_array: SectionArray,
    ):
        self.nodes = section_array_to_nodes(section_array)

        # TODO: fix this
        sagging_temperature = reduce_to_span(
            (section_array.data.sagging_temperature.to_numpy())
        )
        parameter = reduce_to_span(
            section_array.data.sagging_parameter.to_numpy()
        )
        self.span_model = arrays_to_span_model(
            section_array, cable_array, CatenarySpan
        )
        self.balance_model = BalanceModel(
            sagging_temperature,
            self.nodes,
            parameter,
            cable_array,
            self.span_model,
        )

    def solve_adjustment(self):
        self.balance_model.adjustment = True

        sb = Solver()
        sb.solve(self.balance_model)

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
        span_model: Type[Span],
    ):
        # tempertaure and parameter size n-1 here
        self.sagging_temperature = sagging_temperature
        self.nodes = nodes
        self._parameter = parameter
        self.cable_array = cable_array
        # TODO: temporary solution to manage cable data, must find a better way to do this
        self.cable_section = np.float64(self.cable_array.data.section)
        self.diameter = np.float64(self.cable_array.data.diameter)
        self.linear_weight = np.float64(self.cable_array.data.linear_weight)
        self.young_modulus = np.float64(self.cable_array.data.young_modulus)
        self.dilatation_coefficient = np.float64(
            self.cable_array.data.dilatation_coefficient
        )
        self.span_model = span_model
        self.cable_loads: CableLoads = CableLoads(
            self.diameter,
            self.linear_weight,
            np.zeros_like(nodes.weight_chain),
            np.zeros_like(nodes.weight_chain),
        )

        self.adjustment: bool = True
        # TODO: during adjustment computation, perhaps set cable_temperature = 0
        # temperature here is tuning temperature / only in the real span part
        # there is another temperature : change state
        self.nodes.has_load = False

        self.update()
        self.nodes.compute_dx_dy_dz()
        self.nodes.vector_projection.set_tensions(
            self.Th, self.Tv_d, self.Tv_g
        )
        self.nodes.compute_moment()

        self.model_load = LoadModel(
            self.cable_array,
            self.nodes.load,
            self.nodes.load_position,
        )

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
        # TODO: link to mph + decide how to organize Span/Deformation
        self.update_span()

        a = self.a
        b = self.b
        parameter = self.parameter

        cable_length = f.L(
            parameter,
            f.x_n(a, b, parameter),
            f.x_m(a, b, parameter),
        )

        T_mean = f.T_moy(
            p=parameter,
            L=cable_length,
            x_n=f.x_n(a, b, parameter),
            x_m=f.x_m(a, b, parameter),
            linear_weight=self.linear_weight,
        )

        L_ref = cable_length / (
            1
            + self.dilatation_coefficient * self.sagging_temperature
            + T_mean / self.young_modulus / self.cable_section
        )

        self.L_ref = L_ref
        return L_ref

    @property
    def a_prime(self):
        return (self.a**2 + self.b**2 * np.sin(self.beta) ** 2) ** 0.5

    @property
    def b_prime(self):
        return self.b * np.cos(self.beta)

    def compute_Th_and_extremum(self):
        """In this method, a, b, and L_ref may refer to a semi span"""
        parameter = self.approx_parameter(
            self.a_prime, self.b_prime, self.L_ref, self.sagging_temperature
        )
        parameter = find_parameter_function(
            parameter,
            self.a_prime,
            self.b_prime,
            self.L_ref,
            self.sagging_temperature,
            self.k_load,
            self.cable_section,
            self.linear_weight,
            self.dilatation_coefficient,
            self.young_modulus,
        )
        self.span_model.sagging_parameter = fill_to_support(parameter)
        Th = reduce_to_span(self.span_model.T_h())
        x_m = reduce_to_span(self.span_model.x_m())
        x_n = reduce_to_span(self.span_model.x_n())
        return Th, x_m, x_n, parameter

    def update_tensions(self):
        if self.adjustment:
            # Case: adjustment

            Th = reduce_to_span(self.span_model.T_h())

            x_m = reduce_to_span(self.span_model.x_m())
            x_n = reduce_to_span(self.span_model.x_n())

        else:
            # Case: change state + no load:
            Th, x_m, x_n, parameter = self.compute_Th_and_extremum()

            if self.nodes.has_load and abs(np.sum(self.nodes.load)) > 0:
                # Case: change state + with load
                self.x_i = self.nodes.load_position * self.a_prime
                # Don't know why adding x_m is needed
                z_i_m = self.span_model.z_one_point(
                    fill_to_support(self.x_i + x_m)
                )
                z_m = self.span_model.z_one_point(fill_to_support(x_m))
                self.z_i = reduce_to_span(z_i_m - z_m)

                self.solve_xi_zi_loads()

                # Left Th and right Th are equal because balance just got solved (same for parameter)
                a_left = self.x_i
                b_left = self.z_i
                L_ref_left = self.nodes.load_position * self.L_ref

                parameter = self.approx_parameter(
                    a_left, b_left, L_ref_left, self.sagging_temperature
                )
                parameter = find_parameter_function(
                    parameter,
                    a_left,
                    b_left,
                    L_ref_left,
                    self.sagging_temperature,
                    self.k_load,
                    self.cable_section,
                    self.linear_weight,
                    self.dilatation_coefficient,
                    self.young_modulus,
                )
                Th = parameter * self.linear_weight * self.k_load
                x_m_left = f.x_m(a_left, b_left, parameter)

                a_right = self.a_prime - self.x_i
                b_right = self.b_prime - self.z_i
                x_n_right = f.x_n(a_right, b_right, parameter)

                x_m = x_m_left
                x_n = x_n_right

            self._parameter = parameter
            self.span_model.sagging_parameter = fill_to_support(parameter)

        self.Tv_g = Th * (np.sinh(x_m / self._parameter))
        self.Tv_d = -Th * (np.sinh(x_n / self._parameter))
        self.Th = Th

        self.update_projections()

        return Th

    def solve_xi_zi_loads(self):
        self.model_load.set_all(
            self.L_ref,
            self.x_i,
            self.z_i,
            self.a_prime,
            self.b_prime,
            self.k_load,
            self.sagging_temperature,
        )

        solver = Solver()
        solver.solve(
            self.model_load,
            perturb=0.001,
            relax_ratio=0.5,
            stop_condition=1.0,
            relax_power=3,
        )
        self.x_i = self.model_load.x_i
        self.z_i = self.model_load.z_i

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

    @property
    def parameter(self):
        return self._parameter

    def update_span(self):
        """transmet_portee"""
        # warning for dev : we dont use the first element of span vectors for the moment
        z = self.nodes.z

        self.compute_inter()

        self.a = (self.inter1**2 + self.inter2**2) ** 0.5
        b = np.roll(z, -1) - z
        self.b = b[:-1]
        # TODO: fix array lengths?
        self.span_model.span_length = fill_to_support(self.a_prime)
        self.span_model.elevation_difference = fill_to_support(self.b_prime)

    def objective_function(self):
        """[Mx0, My0, Mx1, My1,...]"""
        # Need to run update() before this method if necessary
        self.nodes.compute_moment()

        out = np.array([self.nodes.Mx, self.nodes.My]).flatten('F')
        return out

    @property
    def state_vector(self):
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
        self._load = load
        self.load_position = load_position
        self.has_load = False

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

    @property
    def load(self):
        return self._load

    @load.setter
    def load(self, value):
        self._load = value

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
        L_chain = self.L_chain

        suspension_shift = -((L_chain**2 - self.dx**2 - self.dy**2) ** 0.5)
        self.dz[1:-1] = suspension_shift[1:-1]

        anchor_shift = (L_chain**2 - self.dz**2 - self.dy**2) ** 0.5
        self.dx[0] = anchor_shift[0]
        self.dx[-1] = -anchor_shift[-1]

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
    ):
        self.cable_array = cable_array
        self.cable_section = np.float64(self.cable_array.data.section)
        self.diameter = np.float64(self.cable_array.data.diameter)
        self.linear_weight = np.float64(self.cable_array.data.linear_weight)
        self.young_modulus = np.float64(self.cable_array.data.young_modulus)
        self.dilatation_coefficient = np.float64(
            self.cable_array.data.dilatation_coefficient
        )
        self.load = load
        self.load_position = load_position

    def set_all(
        self,
        L_ref: np.ndarray,
        x_i: np.ndarray,
        z_i: np.ndarray,
        a_prime: np.ndarray,
        b_prime: np.ndarray,
        k_load: np.ndarray,
        temperature,
    ):
        self.L_ref = L_ref
        self.x_i = x_i
        self.z_i = z_i
        self.a_prime = a_prime
        self.b_prime = b_prime
        self.k_load = k_load
        self.temperature = temperature

    @property
    def state_vector(self):
        return np.array([self.x_i, self.z_i]).flatten('F')

    @state_vector.setter
    def state_vector(self, value):
        self.x_i = value[::2]
        self.z_i = value[1::2]

    def objective_function(self):
        # left side of the load
        L_ref_left = self.load_position * self.L_ref
        a_left = self.x_i
        b_left = self.z_i

        Th_left, _, x_n_left, parameter_left = self.compute_Th_and_extremum(
            a_left, b_left, L_ref_left
        )
        Tv_d_loc = -Th_left * np.sinh(x_n_left / parameter_left)

        # right
        L_ref_right = self.L_ref - L_ref_left
        a_right = self.a_prime - self.x_i
        b_right = self.b_prime - self.z_i

        Th_right, x_m_right, _, parameter_right = self.compute_Th_and_extremum(
            a_right, b_right, L_ref_right
        )
        Tv_g_loc = Th_right * np.sinh(x_m_right / parameter_right)

        Th_diff = Th_left - Th_right
        Tv_diff = Tv_d_loc + Tv_g_loc - self.load * self.k_load

        return np.array([Th_diff, Tv_diff]).flatten('F')

    def compute_Th_and_extremum(self, a, b, L_ref):
        """In this method, a, b, and L_ref may refer to a semi span"""
        parameter = self.approx_parameter(a, b, L_ref, self.temperature)
        parameter = find_parameter_function(
            parameter,
            a,
            b,
            L_ref,
            self.temperature,
            self.k_load,
            self.cable_section,
            self.linear_weight,
            self.dilatation_coefficient,
            self.young_modulus,
        )
        Th = parameter * self.linear_weight * self.k_load
        x_m = f.x_m(a, b, parameter)
        x_n = f.x_n(a, b, parameter)
        return Th, x_m, x_n, parameter

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
        """There is nothing to update in this class. Method used in other implementation of ModelForSolver."""
        pass
