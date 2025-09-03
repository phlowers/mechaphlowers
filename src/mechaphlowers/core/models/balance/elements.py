from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import numpy as np
import pandas as pd

from mechaphlowers.core.models.balance import numeric
import mechaphlowers.core.models.balance.functions as f
from mechaphlowers.core.models.balance.utils_balance import MapVectorToNodeSpace, VectorProjection
from mechaphlowers.core.models.external_loads import CableLoads
import mechaphlowers.core.numeric.numeric as optimize


@dataclass
class Cable:
    section: np.ndarray  # input in mm ?
    lineic_weight: np.ndarray
    dilation_coefficient: np.ndarray
    young_modulus: np.ndarray
    diameter: np.ndarray
    cra: np.ndarray


class Span:
    def __init__(
        self,
        sagging_temperature: np.ndarray,
        nodes: Nodes,
        parameter: np.ndarray = None,
        cable: Cable = None,
        adjustment: bool = True,
    ):
        self.sagging_temperature = sagging_temperature
        self.adjustment = adjustment
        self.nodes = nodes
        self._parameter = parameter * np.ones(len(self.nodes) - 1)
        self.cable = cable
        self.cable_loads: CableLoads = CableLoads(
            cable.diameter,
            cable.lineic_weight,
            np.zeros_like(cable.diameter),
            np.zeros_like(cable.diameter),
        )

        self.nodes.no_load = True
        self.init_Lref_mode = False
        # TODO: during L_ref computation, perhaps set cable_temperature = 0
        # temperature here is tuning temperature / only in the real span part
        # there is another temperature : change state

        self.update_span()
        self.nodes.vector_projection.set_tensions(
            self.Th, self.Tv_d, self.Tv_g
        )
        self.nodes.compute_forces(True)

    def adjust(self):
        self.nodes.no_load = False

        SolverBalance().solver_balance_3d(self)

        self._L_ref = RealSpan(self).real_span()

    @property
    def alpha(self):
        beta = self.cable_loads.load_angle
        return (
            np.acos(
                self.a / (self.a**2 + self.b**2 * np.sin(beta) ** 2) ** 0.5
            )
            * np.sign(beta)
            * np.sign(self.b)
        )

    @property
    def L_ref(self):
        return self._L_ref

    def update_length(self):
        RealSpan(self).real_span()

    @property
    def Th(self):
        self.update_tensions()
        return self._Th

    def update_tensions(self):
        if not self.adjustment:
            c_param = self.cardan(
                self.a, self.b, self.L_ref, self.sagging_temperature
            )

            c_param = self.find_parameter(
                c_param,
                self.a,
                self.b,
                self.L_ref,
                self.sagging_temperature,
            )
            self._parameter = c_param
        else:
            c_param = self._parameter

        Th = c_param * self.cable.lineic_weight * np.ones(len(self.nodes) - 1)

        x_m = f.x_m(self.a, self.b, c_param)
        x_n = f.x_n(self.a, self.b, c_param)

        self._Tv_g = Th * (np.sinh(x_m / c_param))
        self._Tv_d = -Th * (np.sinh(x_n / c_param))
        self._Th = Th

        self.update_projections()

        return Th

    def update_projections(self):
        # TODO: alpha and beta
        alpha = self.alpha
        beta = self.cable_loads.load_angle

        self.compute_inter()
        self.nodes.vector_projection.set_all(
            self._Th,
            self._Tv_d,
            self._Tv_g,
            alpha,
            beta,
            self.nodes.line_angle,
            self.proj_angle,
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

    def get_approximative_parameter(self):
        self._parameter = self.cardan(
            self.a, self.b, self.L_ref, self.sagging_temperature
        )

    def cardan(self, a, b, L0, cable_temperature):
        circle_chord = (a**2 + b**2) ** 0.5

        factor = (
            self.cable.lineic_weight
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

    def find_parameter(self, parameter, a, b, L0, cable_temperature):
        """this is a placehoder of sagtension algorithm"""
        param = parameter

        n_iter = 50

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
            )

            delta1 = (lon - self.L_ref) / self.L_ref - (
                self.cable.dilation_coefficient * self.sagging_temperature
                + Tm1 / (self.cable.young_modulus) / self.cable.section
            )

            mem = param
            param = param + 1

            x_m = f.x_m(a, b, param)
            x_n = f.x_n(a, b, param)
            lon = f.L(param, x_n, x_m)
            Tm1 = f.T_moy(
                p=param,
                L=lon,
                x_n=a + f.x_m(a, b, param),
                x_m=f.x_m(a, b, param),
                lineic_weight=self.cable.lineic_weight,
            )

            delta2 = (lon - self.L_ref) / self.L_ref - (
                self.cable.dilation_coefficient * self.sagging_temperature
                + Tm1 / (self.cable.young_modulus) / self.cable.section
            )

            param = (param - 1) - delta1 / (delta2 - delta1)

            if np.linalg.norm(mem - param) < 0.1 * param.size:
                break
            if i == n_iter:
                print("max iter reached")

        return param

    def update_span(self):
        """transmet_portee"""
        # warning for dev : we dont use the first element of span vectors for the moment
        # if self.init_Lref_mode is True:
        #     x = self.nodes._x
        #     z = self.nodes._z
        # else:
        x = self.nodes.x
        z = self.nodes.z

        self.compute_inter()

        self.a = (self.inter1**2 + self.inter2**2) ** 0.5
        b = z - np.roll(z, 1)
        self.b = b[1:]


    def vector_force(self, update_dx_dz=True):
        self.nodes.vector_projection.set_tensions(
            self.Th,
            self.Tv_d,
            self.Tv_g,
        )
        self.nodes.compute_forces(
            update_dx_dy_dz=update_dx_dz,
        )

        out = np.array([self.nodes.Mx, self.nodes.My]).flatten('F')
        return out


    def _delta_dz(self, dz):
        self.nodes.dz += dz
        self.nodes.compute_dx_dy_dz()
        self.update_span()  # transmet_portee: update a and b

        self.update_tensions()  # Th : cardan for parameter then compute Th, Tvd, Tvg

        force_vector = self.vector_force(update_dx_dz=False)

        self.nodes.dz -= dz

        self.nodes.compute_dx_dy_dz()
        # TODO: coherence with _delta_dx after other usecases
        # TODO: here the following steps have been removed but the span object is not set in the same state.
        # self.update_span()
        # self.update_tensions()

        return force_vector

    def _delta_dy(self, dy):
        self.nodes.dy += dy
        # self.update_span()
        self.nodes.compute_dx_dy_dz()
        self.update_tensions()

        force_vector = self.vector_force(update_dx_dz=False)

        self.nodes.dy -= dy
        # TODO: here the following steps have been removed but the span object is not set in the same state.
        self.nodes.compute_dx_dy_dz()
        # self.update_span()
        # self.update_tensions()

        return force_vector


    def _delta_dx(self, dx):
        self.nodes.dx += dx
        # self.update_span()
        self.update_span()
        self.nodes.compute_dx_dy_dz()
        self.update_tensions()

        force_vector = self.vector_force(update_dx_dz=False)

        self.nodes.dx -= dx
        # TODO: here the following steps have been removed but the span object is not set in the same state.
        self.nodes.compute_dx_dy_dz()
        # self.update_span()
        # self.update_tensions()

        return force_vector

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


class RealSpan:
    def __init__(self, span: Span):
        self.span = span

    def real_span(self):
        # TODO: this part take the hypothesis that there is one of two node of ntype 1.
        # we should change this to be more general

        a = np.roll(self.span.nodes.x, -1) - np.roll(
            self.span.nodes.x, 1
        )
        b = np.roll(self.span.nodes.z, -1) - np.roll(
            self.span.nodes.z, 1
        )

        parameter_np1 = np.hstack(
            (self.span.parameter, self.span.parameter[-1])
        )

        lon1 = f.L(
            parameter_np1,
            a + f.x_m(a, b, parameter_np1),
            f.x_m(a, b, parameter_np1),
        )

        Tm1 = f.T_moy(
            p=parameter_np1,
            L=lon1,
            x_n=a + f.x_m(a, b, parameter_np1),
            x_m=f.x_m(a, b, parameter_np1),
            lineic_weight=self.span.cable.lineic_weight,
        )

        lon1 = lon1 / (
            1
            + self.span.cable.dilation_coefficient
            * self.span.sagging_temperature
            + Tm1 / self.span.cable.young_modulus / self.span.cable.section
        )

        pos_charge = (
            self.span.nodes.x
            - np.roll(self.span.nodes.x, 1)
            - np.roll(self.span.nodes.dx, 1)
        )

        lon2 = f.L(
            parameter_np1,
            pos_charge + f.x_m(a, b, parameter_np1),
            f.x_m(a, b, parameter_np1),
        )

        Tm2 = f.T_moy(
            p=parameter_np1,
            x_n=pos_charge + f.x_m(a, b, parameter_np1),
            x_m=f.x_m(a, b, parameter_np1),
            L=lon2,
            lineic_weight=self.span.cable.lineic_weight,
        )

        lon2 = lon2 / (
            1
            + self.span.cable.dilation_coefficient
            * self.span.sagging_temperature
            + Tm2 / self.span.cable.young_modulus / self.span.cable.section
        )

        # we need np.array([lon2[1], lon1-lon2[1], lon2[3], ...])
        L_ref = np.reshape(
            np.vstack(  # stacking the two array vertically and taking only 1/2 node
                (lon2[1::2], lon1[1::2] - lon2[1::2])
            ),
            -1,
            order='F',
        )  # order='F' is for fortran order to flatten the array to get the good form

        return L_ref


class Masks:
    def __init__(self, ntype):
        # first and last support are supposed to be clamped
        self._ntype = ntype
        self._ntype[-1] = 4

    def mask(self, ntype):
        return self._ntype == ntype

    def filter(self, vector, ntype):
        return np.where(self._ntype == ntype, vector, 0)


class Nodes:
    def __init__(
        self,
        ntype: np.ndarray,
        L_chain: np.ndarray,
        weight_chain: np.ndarray,
        arm_length: np.ndarray,
        line_angle: np.ndarray,
        x: np.ndarray,
        z: np.ndarray,
        load: np.ndarray,
    ):
        self.num = np.arange(len(ntype))
        self.ntype = ntype
        self.L_chain = L_chain
        self.weight_chain = -weight_chain
        # arm length: positive length means further from observer
        self.arm_length = arm_length
        # line_angle: anti clockwise
        self.line_angle = line_angle
        self.init_coordinates(x, z)
        # dx, dy, dz are the distances between the, including the chain
        self.dx = np.zeros_like(x, dtype=np.float64)
        self.dy = np.zeros_like(x, dtype=np.float64)
        self.dz = np.zeros_like(z, dtype=np.float64)
        self._load = -load
        self.no_load = False

        self.vector_projection = VectorProjection()

        self.mask = Masks(ntype)

    @property
    def load(self):
        if self.no_load is True:
            return np.zeros_like(self._load)
        return self._load

    @load.setter
    def load(self, value):
        self._load = value

    def __len__(self):
        return len(self.num)


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
        L = self.L_chain

        suspension_shift = -(L**2 - self.dx**2 - self.dy**2) ** 0.5
        self.dz[1:-1] = suspension_shift[1:-1]
        
        anchor_shift = (L**2 - self.dz**2 - self.dy**2) ** 0.5
        self.dx[0] = anchor_shift[0]
        self.dx[-1] = -anchor_shift[-1]
        

    def compute_forces(self, update_dx_dy_dz=True):
        # Placeholder for force computation logic

        if update_dx_dy_dz:
            self.compute_dx_dy_dz()


        s_right, t_right, z_right = self.vector_projection.T_line_plane_right()
        T_line_plane_left = self.vector_projection.T_line_plane_left()
        s_left, t_left, z_left = T_line_plane_left
        s_left_rolled, t_left_rolled, z_left_rolled = np.roll(T_line_plane_left, -1, axis=1)

        gamma = (self.line_angle / 2) [1:]

        # Not entierly sure about indices and left/right

        # index 1 ou 0?
        Fx_first = s_left[0] * np.cos((self.line_angle / 2)[0]) - t_left[0] * np.sin((self.line_angle / 2)[0])
        Fy_first = t_left[0] * np.cos((self.line_angle / 2)[0]) + s_left[0] * np.sin((self.line_angle / 2)[0])
        Fz_first = z_left[0] + self.weight_chain[0] / 2 # also add load?


        Fx_suspension = (s_right + s_left_rolled) * np.cos(gamma) - (-t_right + t_left_rolled) * np.sin(gamma)
        Fy_suspension = (t_right + t_left_rolled) * np.cos(gamma) - (s_right - s_left_rolled) * np.sin(gamma)
        Fz_suspension = z_right + z_left_rolled  + self.weight_chain[1:] / 2

        Fx_last = (s_right[-1]) * np.cos(gamma[-1]) - (-t_right[-1]) * np.sin(gamma[-1])
        Fy_last = (t_right[-1]) * np.cos(gamma[-1]) - (s_right[-1]) * np.sin(gamma[-1])
        Fz_last = z_right[-1] + self.weight_chain[-1] / 2

        Fx = np.concat(([Fx_first], Fx_suspension[:-1], [Fx_last]))
        Fy = np.concat(([Fy_first], Fy_suspension[:-1], [Fy_last]))
        Fz = np.concat(([Fz_first], Fz_suspension[:-1], [Fz_last]))

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
            'num': self.num,
            'ntype': self.ntype,
            'L_chain': self.L_chain,
            'weight_chain': self.weight_chain,
            'x': self.x,
            'z': self.z_arm,
            'dx': self.dx,
            'dz': self.dz,
            'load': self.load,
            'Fx': self.Fx,
            'Fz': self.Fz,
            'My': self.My,
        }
        out = pd.DataFrame(data)

        return str(out)

    def __str__(self):
        return self.__repr__()


class SolverBalance:
    def __init__(self, adjustment=True):
        self.mem_loop = []
        self.adjustment = adjustment

    def solver_balance_3d(
        self,
        section: Span,
        temperature=0,
    ):
        puissance = 3
        
        section.update_tensions()
        section.update_span()
        
        # initialisation
        perturb = 0.0001
        force_vector = section.vector_force()
        n_iter = range(1, 100)
        vector_perturb = np.zeros_like(section.nodes.dx)
        # for debug we record init_force
        self.init_force = force_vector
        relaxation = .5
        
        
        # starting optimisation loop
        for compteur in n_iter:
            # compute jacobian
            df_list = []

            for i in range(len(section.nodes.ntype)):
                vector_perturb[i] += perturb

                # TODO: refactor if/elif ? + node logic should not be in the solver
                if i == 0 or i == len(section.nodes.ntype):
                    dz_d = section._delta_dz(vector_perturb)
                    dF_dz = (dz_d - force_vector) / perturb
                    df_list.append(dF_dz)
                    
                    dy_d = section._delta_dy(vector_perturb)
                    dF_dy = (dy_d - force_vector) / perturb
                    df_list.append(dF_dy)
                    
                    vector_perturb[i] -= perturb

                else:
                    dx_d = section._delta_dx(vector_perturb)
                    dF_dx = (dx_d - force_vector) / perturb
                    df_list.append(dF_dx)
                    
                    dy_d = section._delta_dy(vector_perturb)
                    dF_dy = (dy_d - force_vector) / perturb
                    df_list.append(dF_dy)
                    
                    vector_perturb[i] -= perturb
                    
            jacobian = np.array(df_list)

            # memorize for norm
            mem = np.linalg.norm(force_vector)

            # correction calculus
            # TODO: check the cross product matrix / vector
            correction = np.linalg.inv(jacobian.T) @ force_vector
            
            correction_mx = correction[::2]
            correction_my = correction[1::2]
            
            section.nodes.dx[1:-1] = section.nodes.dx[1:-1] - correction_mx[1:-1] * (1 - relaxation ** (compteur ** puissance))
            section.nodes.dy[1:-1] = section.nodes.dy[1:-1] - correction_my[1:-1] * (1 - relaxation ** (compteur ** puissance))
            section.nodes.dz[1:-1] = -(section.nodes.L_chain[1:-1] ** 2 - section.nodes.dx[1:-1] ** 2 - section.nodes.dy[1:-1] ** 2) ** 0.5
            
            section.nodes.dz[[0,-1]] = section.nodes.dz[[0,-1]] - correction_mx[[0,-1]] * (1 - relaxation ** (compteur ** puissance))
            section.nodes.dy[[0,-1]] = section.nodes.dy[[0,-1]] - correction_my[[0,-1]] * (1 - relaxation ** (compteur ** puissance))
            section.nodes.dx[0] = (section.nodes.L_chain[0] ** 2 - section.nodes.dz[0] ** 2 - section.nodes.dy[0] ** 2) ** 0.5
            section.nodes.dx[-1] = -(section.nodes.L_chain[-1] ** 2 - section.nodes.dz[-1] ** 2 - section.nodes.dy[-1] ** 2) ** 0.5

            
            # update
            section.nodes.compute_dx_dy_dz()
            section.update_tensions()
            section.update_span()

            # compute value to minimize
            force_vector = section.vector_force()
            norm_d_param = np.abs(np.linalg.norm(force_vector) ** 2 - mem**2)

            print("**" * 10)
            print(compteur)
            # print(correction[1:-1])
            print("force vector norm: ", np.linalg.norm(force_vector) ** 2)
            print(f"{norm_d_param=}")
            # print("-"*10)
            # print(section.nodes.dx)
            # print(section.nodes.dz)

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
            if norm_d_param < 0.1:
                # print("--end--"*10)
                # print(norm_d_param)
                break
            if n_iter == compteur:
                print("max iteration reached")
                print(norm_d_param)

        print(f"force vector norm: {np.linalg.norm(force_vector)}")
        self.final_dx = section.nodes.dx
        self.final_dz = section.nodes.dz
        self.final_force = force_vector

        