# Copyright (c) 2025, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0


from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

import mechaphlowers.core.models.balance.functions as f
from mechaphlowers.core.models.balance import numeric
from mechaphlowers.core.models.balance.utils_balance import (
    VectorProjection,
)
from mechaphlowers.core.models.external_loads import CableLoads


@dataclass
class Cable:
    section: np.float64  # input in mm ?
    lineic_weight: np.float64
    dilation_coefficient: np.float64
    young_modulus: np.float64
    diameter: np.float64
    cra: np.float64


class Span:
    def __init__(
        self,
        sagging_temperature: float,
        nodes: Nodes,
        parameter: float,
        cable: Cable,
    ):
        self.sagging_temperature = np.float64(sagging_temperature)
        self.nodes = nodes
        self._parameter = parameter * np.ones(len(self.nodes) - 1)
        self.cable = cable
        self.cable_loads: CableLoads = CableLoads(
            cable.diameter,
            cable.lineic_weight,
            np.zeros_like(nodes.weight_chain),
            np.zeros_like(nodes.weight_chain),
        )

        self.adjustment = True
        # TODO: during adjustment computation, perhaps set cable_temperature = 0
        # temperature here is tuning temperature / only in the real span part
        # there is another temperature : change state
        self.nodes.has_load = False

        self.update_span()
        self.update_tensions()
        self.nodes.vector_projection.set_tensions(
            self._Th, self.Tv_d, self.Tv_g
        )
        self.nodes.compute_moment()

    def adjust(self):
        self.adjustment = True

        self.sb = SolverBalance()
        self.sb.solver_balance_3d(self)

        self._L_ref = self.update_L_ref()

    def change_state(self):
        self.adjustment = False
        self.nodes.has_load = True

        self.sb = SolverBalance()
        self.sb.solver_balance_3d(self)

    @property
    def k_load(self):
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
        # Sign different from what already exists in CableLoads
        return -self.cable_loads.load_angle[0:-1]

    @property
    def L_ref(self):
        return self._L_ref

    def update_L_ref(self):
        self.update_span()

        a = self.a
        b = self.b
        parameter = self.parameter

        cable_length = f.L(
            parameter,
            a + f.x_m(a, b, parameter),
            f.x_m(a, b, parameter),
        )

        T_mean = f.T_moy(
            p=parameter,
            L=cable_length,
            x_n=a + f.x_m(a, b, parameter),
            x_m=f.x_m(a, b, parameter),
            lineic_weight=self.cable.lineic_weight,
        )

        L_ref = cable_length / (
            1
            + self.cable.dilation_coefficient * self.sagging_temperature
            + T_mean / self.cable.young_modulus / self.cable.section
        )

        self._L_ref = L_ref
        return L_ref

    @property
    def Th(self):
        return self._Th

    @property
    def a_prime(self):
        return (self.a**2 + self.b**2 * np.sin(self.beta) ** 2) ** 0.5

    @property
    def b_prime(self):
        return self.b * np.cos(self.beta)

    def compute_Th_and_extremum(self, a, b, L_ref):
        """In this method, a, b, and L_ref may refer to a semi span"""
        parameter = self.cardan(a, b, L_ref, self.sagging_temperature)
        parameter = self.find_parameter(
            parameter,
            a,
            b,
            L_ref,
            self.sagging_temperature,
        )
        Th = parameter * self.cable.lineic_weight * self.k_load
        x_m = f.x_m(a, b, parameter)
        x_n = f.x_n(a, b, parameter)
        return Th, x_m, x_n, parameter

    def update_tensions(self):
        if self.adjustment:
            # Case: adjustment
            parameter = self._parameter

            Th = parameter * self.cable.lineic_weight * self.k_load

            x_m = f.x_m(self.a_prime, self.b_prime, parameter)
            x_n = f.x_n(self.a_prime, self.b_prime, parameter)

        else:
            # Case: change state + no load:
            Th, x_m, x_n, parameter = self.compute_Th_and_extremum(
                self.a_prime, self.b_prime, self.L_ref
            )
            self._parameter = parameter

            if self.nodes.has_load and abs(np.sum(self.nodes.load)) > 0:
                # Case: change state + with load
                self.x_i = self.nodes.load_position * self.a_prime
                self.z_i = f.z(self.x_i, self.parameter, x_m) - f.z(
                    np.zeros_like(self.x_i), self.parameter, x_m
                )
                self.compute_tensions_with_loads()

                # Left Th and right Th are equal because balance just got solved (same for parameter)
                a_left = self.x_i
                b_left = self.z_i
                L_ref_left = self.nodes.load_position * self.L_ref
                Th, x_m_left, _, parameter = self.compute_Th_and_extremum(
                    a_left, b_left, L_ref_left
                )

                a_right = self.a_prime - self.x_i
                b_right = self.b_prime - self.z_i
                x_n_right = f.x_n(a_right, b_right, parameter)

                x_m = x_m_left
                x_n = x_n_right

        self._Tv_g = Th * (np.sinh(x_m / parameter))
        self._Tv_d = -Th * (np.sinh(x_n / parameter))
        self._Th = Th

        self.update_projections()

        return Th

    def compute_tensions_with_loads(self):
        solver_load = SolverLoad()
        solver_load.solver_tensions_load(self)

    def update_projections(self):
        alpha = self.alpha
        beta = self.beta

        self.compute_inter()
        self.nodes.vector_projection.set_all(
            self._Th,
            self._Tv_d,
            self._Tv_g,
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
        span_length = np.ediff1d(self.nodes.x_arm)
        self.inter1 = span_length + proj_diff[0]
        self.inter2 = proj_diff[1]
        self.proj_angle = np.atan2(self.inter2, self.inter1)

    @property
    def Tv_g(self):
        return self._Tv_g

    @property
    def Tv_d(self):
        return self._Tv_d

    def cardan(self, a, b, L0, cable_temperature):
        circle_chord = (a**2 + b**2) ** 0.5

        factor = (
            self.cable.lineic_weight
            * self.k_load
            / self.cable.young_modulus
            / self.cable.section
        )

        p3 = factor * L0
        p2 = (
            L0
            - circle_chord
            + self.cable.dilation_coefficient * cable_temperature * L0
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

    def find_parameter(self, parameter, a, b, L_ref, cable_temperature):
        """this is a placeholder of sagtension algorithm"""
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
                x_n=a + f.x_m(a, b, param),
                x_m=f.x_m(a, b, param),
                lineic_weight=self.cable.lineic_weight,
                k_load=self.k_load,
            )

            delta1 = (lon - L_ref) / L_ref - (
                self.cable.dilation_coefficient * self.sagging_temperature
                + Tm1 / (self.cable.young_modulus) / self.cable.section
            )

            mem = param
            param = param + step

            x_m = f.x_m(a, b, param)
            x_n = f.x_n(a, b, param)
            lon = f.L(param, x_n, x_m)
            Tm1 = f.T_moy(
                p=param,
                L=lon,
                x_n=a + f.x_m(a, b, param),
                x_m=f.x_m(a, b, param),
                lineic_weight=self.cable.lineic_weight,
                k_load=self.k_load,
            )

            delta2 = (lon - L_ref) / L_ref - (
                self.cable.dilation_coefficient * self.sagging_temperature
                + Tm1 / (self.cable.young_modulus) / self.cable.section
            )

            param = (param - step) - delta1 / (delta2 - delta1)

            if np.linalg.norm(mem - param) < 0.1 * param.size:
                break
            if i == n_iter:
                print("max iter reached")

        return param

    def update_span(self):
        """transmet_portee"""
        # warning for dev : we dont use the first element of span vectors for the moment
        z = self.nodes.z

        self.compute_inter()

        self.a = (self.inter1**2 + self.inter2**2) ** 0.5
        b = np.roll(z, -1) - z
        self.b = b[:-1]

    def vector_moment(self):
        """[Mx0, My0, Mx1, My1,...]"""
        self.update_tensions()
        self.nodes.vector_projection.set_tensions(
            self._Th,
            self.Tv_d,
            self.Tv_g,
        )
        self.nodes.compute_moment()

        out = np.array([self.nodes.Mx, self.nodes.My]).flatten('F')
        return out

    def local_tension_matrix(self):
        # left side of the load
        L_ref_left = self.nodes.load_position * self.L_ref
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

        Th = Th_left - Th_right
        Tv = Tv_d_loc + Tv_g_loc - self.nodes.load * self.k_load

        return np.array([Th, Tv]).flatten('F')
    
    @property
    def state_vector(self):
        return self.nodes.state_vector
    
    @state_vector.setter
    def state_vector(self, value):
        self.nodes.state_vector = value
    
    def jacobian(self, force_vector, perturb=1e-4):
        
        vector_perturb = np.zeros_like(force_vector)
        df_list = []

        for i in range(len(vector_perturb)):
            vector_perturb[i] += perturb

            M_perturb = self._delta_d(vector_perturb)
            dM_dperturb = (M_perturb - force_vector) / perturb
            df_list.append(dM_dperturb)

            vector_perturb[i] -= perturb

        jacobian = np.array(df_list)
        return jacobian
    

    def _delta_d(self, perturbation):

        
        self.state_vector += perturbation
        
        self.update()
        # self.nodes.compute_dx_dy_dz()
        # self.update_tensions()  # Th : cardan for parameter then compute Th, Tvd, Tvg
        # self.update_span()  # transmet_portee: update a and b

        force_vector = self.vector_moment()
        self.state_vector -= perturbation



        self.nodes.compute_dx_dy_dz()
        # TODO: here the following steps have been removed but the span object is not set in the same state.
        # self.update_span()
        # self.update_tensions()

        return force_vector

    def _delta_d_load(
        self, perturbation, variable_name: Literal["x_i", "z_i"]
    ):
        self.__dict__[variable_name] += perturbation

        force_vector = self.local_tension_matrix()
        self.__dict__[variable_name] -= perturbation

        return force_vector
    
    def update(self):
        # self.nodes.compute_dx_dy_dz()
        self.update_tensions()
        self.update_span()

    def __repr__(self):
        data = {
            'parameter': self.parameter,
            'cable_temperature': self.sagging_temperature,
            'Th': self._Th,
            'Tv_d': self._Tv_d,
            'Tv_g': self._Tv_g,
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
    cable_temperature: np.ndarray,
    k_load: np.ndarray,
    cable_section: np.float64,
    lineic_weight: np.float64,
    dilation_coefficient: np.float64,
    young_modulus: np.float64,
):
    """this is a placeholder of sagtension algorithm"""
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
            x_n=a + f.x_m(a, b, param),
            x_m=f.x_m(a, b, param),
            lineic_weight=lineic_weight,
            k_load=k_load,
        )

        delta1 = (lon - L_ref) / L_ref - (
            dilation_coefficient * cable_temperature
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
            x_n=a + f.x_m(a, b, param),
            x_m=f.x_m(a, b, param),
            lineic_weight=lineic_weight,
            k_load=k_load,
        )

        delta2 = (lon - L_ref) / L_ref - (
            dilation_coefficient * cable_temperature
            + Tm1 / young_modulus / cable_section
        )

        param = (param - step) - delta1 / (delta2 - delta1)

        if np.linalg.norm(mem - param) < 0.1 * param.size:
            break
        if i == n_iter:
            print("max iter reached")

    return param




class Nodes:
    def __init__(
        self,
        L_chain: np.ndarray,
        weight_chain: np.ndarray,
        arm_length: np.ndarray,
        line_angle: np.ndarray,
        x: np.ndarray,
        z: np.ndarray,
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
        self.init_coordinates(x, z)
        # dx, dy, dz are the distances between the, including the chain
        self.dxdydz = np.zeros((3,len(x)), dtype=np.float64)
        # self.dx = 
        # self.dy = np.zeros_like(x, dtype=np.float64)
        # self.dz = np.zeros_like(z, dtype=np.float64)
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
        if self.has_load is False:
            return np.zeros_like(self._load)
        return self._load

    @load.setter
    def load(self, value):
        self._load = value

    def __len__(self):
        return len(self.L_chain)

    @property
    def x(self):
        """This property returns the x coordinate of the end the chain. x = x_arm + dx"""
        return self.x_arm + self.dx

    # useless?
    @property
    def y(self):
        return self._y + self.dy

    @property
    def z(self):
        """This property returns the altitude of the end the chain.  z = z_arm + dz"""
        return self.z_arm + self.dz

    @property
    def x_arm(self):
        """This property returns the x coordinate of the end the arm. Should not be modified during computation."""
        return self._x0

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
        
        dxdy = self.dxdydz[[0,1], 1:-1]
        dzdy = self.dxdydz[[2,1]][:,[0,-1]]
        return np.vstack([dzdy[:,0], dxdy.T, dzdy[:,1]]).flatten()
    
    @state_vector.setter
    def state_vector(self, state_vector):
        #TODO: refactor with mask
        dzdxdz = state_vector[::2]
        dy = state_vector[1::2]
        
        self.dy = dy
        self.dx[1:-1] = dzdxdz[1:-1]
        self.dz[[0,-1]] = dzdxdz[[0,-1]]
        #TODO: verify compute_dxdy_dz is not too much called
        self.compute_dx_dy_dz()

    def init_coordinates(self, x, z):
        self.x_anchor_chain = np.zeros_like(x)
        self.x_anchor_chain[0] = self.L_chain[0]
        self.x_anchor_chain[-1] = -self.L_chain[-1]
        self.z_suspension_chain = np.zeros_like(x)
        self.z_suspension_chain[1:-1] = -self.L_chain[1:-1]

        # warning: x0 and z0 does not mean the same thing
        # x0 is the absissa of the
        # z0 is the altitude of the attachement point
        self._x0 = x
        self._z0 = z
        self._y = np.zeros_like(x, dtype=np.float64)

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

        self.compute_dx_dy_dz()

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

    def debug(self):
        data = {
            'z': self.z,
            'x': self.x,
            'dz': self.dz,
            'Fx': self.Fx,
            'Fz': self.Fz,
            'My': self.My,
        }
        out = pd.DataFrame(data)

        return out

    def __repr__(self):
        data = {
            'L_chain': self.L_chain,
            'weight_chain': self.weight_chain,
            'x': self.x,
            'z': self.z_arm,
            'dx': self.dx,
            'dz': self.dz,
            'Fx': self.Fx,
            'Fz': self.Fz,
            'My': self.My,
        }
        out = pd.DataFrame(data)

        return str(out)

    def __str__(self):
        return self.__repr__()


class SolverLoad:
    def __init__(self):
        self.mem_loop = []

    def solver_tensions_load(
        self,
        section: Span,
    ):
        puissance = 3

        # initialisation
        perturb = 0.001
        tension_vector = section.local_tension_matrix()
        n_iter = range(1, 100)
        vector_perturb = np.zeros_like(section.x_i)
        # for debug we record init_force
        self.init_force = tension_vector
        relaxation = 0.5
        # too strict?
        stop_condition = 1.0

        # starting optimisation loop
        for compteur in n_iter:
            # compute jacobian
            df_list = []

            for i in range(len(section.x_i)):
                vector_perturb[i] += perturb

                dxi_d = section._delta_d_load(vector_perturb, "x_i")
                dT_dxi = (dxi_d - tension_vector) / perturb
                df_list.append(dT_dxi)

                dzi_d = section._delta_d_load(vector_perturb, "z_i")
                dT_dzi = (dzi_d - tension_vector) / perturb
                df_list.append(dT_dzi)

                vector_perturb[i] -= perturb

            jacobian = np.array(df_list)

            # memorize for norm
            mem = np.linalg.norm(tension_vector)

            # correction calculus
            correction = np.linalg.inv(jacobian.T) @ tension_vector

            correction_x_i = correction[::2]
            correction_z_i = correction[1::2]

            section.x_i = section.x_i - correction_x_i * (
                1 - relaxation ** (compteur**puissance)
            )
            section.z_i = section.z_i - correction_z_i * (
                1 - relaxation ** (compteur**puissance)
            )

            # compute value to minimize
            tension_vector = section.local_tension_matrix()
            norm_d_param = np.abs(np.linalg.norm(tension_vector) ** 2 - mem**2)

            print("**" * 10)
            print("Solver x_i/z_i counter: ", compteur)
            # print(correction[1:-1])
            print("tension vector norm: ", np.linalg.norm(tension_vector) ** 2)
            print(f"{norm_d_param=}")
            print("x_i: ", section.x_i)
            print("z_i: ", section.z_i)
            print(f"{norm_d_param=}")
            print("-" * 10)
            print(section.nodes.dx)
            print(section.nodes.dz)

            self.mem_loop.append(
                {
                    "num_loop": compteur,
                    "norm_d_param": norm_d_param,
                    "tension": tension_vector,
                    "correction": correction,
                }
            )

            # check value to minimze to break the loop
            if norm_d_param < stop_condition:
                # print("--end--"*10)
                # print(norm_d_param)
                break
            if n_iter == compteur:
                print("max iteration reached")
                print(norm_d_param)


class SolverBalance:
    def __init__(self, adjustment=True):
        self.mem_loop = []
        self.adjustment = adjustment

    def solver_balance_3d(
        self,
        section: Span,
    ):
        puissance = 3

        section.update()

        # initialisation
        perturb = 0.0001
        force_vector = section.vector_moment()
        n_iter = range(1, 100)
        
        # for debug we record init_force
        self.init_force = force_vector
        relaxation = 0.8
        stop_condition = 1e-3

        # starting optimisation loop
        for compteur in n_iter:
            # compute jacobian
            jacobian = section.jacobian(force_vector, perturb)

            # memorize for norm
            mem = np.linalg.norm(force_vector)

            # correction calculus
            correction = np.linalg.inv(jacobian.T) @ force_vector

            section.state_vector = section.state_vector - correction * (1 - relaxation ** (compteur**puissance))

            # update
            section.update()

            # compute value to minimize
            force_vector = section.vector_moment()
            norm_d_param = np.abs(np.linalg.norm(force_vector) ** 2 - mem**2)

            print("**" * 10)
            print(compteur)
            # print(correction[1:-1])
            print("force vector norm: ", np.linalg.norm(force_vector) ** 2)
            print(f"{norm_d_param=}")
            print("-" * 10)
            print(section.nodes.dx)
            print(section.nodes.dz)

            self.mem_loop.append(
                {
                    "num_loop": compteur,
                    "norm_d_param": norm_d_param,
                    "force": force_vector,
                    "dx": section.nodes.dx,
                    "dz": section.nodes.dz,
                    "correction": correction,
                }
            )

            # check value to minimze to break the loop
            if norm_d_param < stop_condition:
                # print("--end--"*10)
                # print(norm_d_param)
                break
            if n_iter == compteur:
                print("max iteration reached")
                print(norm_d_param)

        print(f"force vector norm: {np.linalg.norm(force_vector)}")
        self.final_dx = section.nodes.dx
        self.final_dy = section.nodes.dy
        self.final_dz = section.nodes.dz
        self.final_force = force_vector


