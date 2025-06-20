from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

import mechaphlowers.core.models.balance.functions as f
import mechaphlowers.core.numeric.numeric as optimize


class NodeType(Enum):
    DDL_2 = 1
    DDL_1V = 2
    DDL_1H = 3

@dataclass
class Cable:
    section: np.ndarray # input in mm ?
    lineic_weight: np.ndarray
    dilation_coefficient: np.ndarray
    young_modulus: np.ndarray


class Span:
    def __init__(
        self,
        # a: np.ndarray,
        # b: np.ndarray,
        # length_0: np.ndarray,
        cable_temperature: np.ndarray,
        nodes: Nodes,
        parameter: np.ndarray = None,
        cable: Cable = None,
        reglage: bool = True,
        finition: bool = False,
    ):
        # self.a
        # self.b
        # self.length_0
        self.cable_temperature = cable_temperature
        self.reglage = reglage
        self.nodes = nodes
        self.finition = finition
        self._parameter = parameter*np.ones(len(self.nodes)-1)
        self.cable = cable
        
        self.update_span()
        self.nodes.compute_forces(self.Th, self.Tv_d, self.Tv_g, self.parameter)
    
    @property
    def Th(self):
        if not self.reglage:
            c_param = self.cardan(self.a, self.b, self.length_0, self.cable_temperature)
            if self.finition:
                c_param = self.find_parameter(self.parameter, self.a, self.b, self.length_0, self.cable_temperature)
        else:
            c_param = self._parameter
            
        Th = c_param * self.cable.lineic_weight * np.ones(len(self.nodes)-1)
        
        x_m = f.x_m(self.a, self.b, c_param)
        x_n = f.x_n(self.a, self.b, c_param)
        
        self._Tv_g = -Th * (np.sinh(x_m / c_param)) # moins ?
        self._Tv_d = Th * (np.sinh(x_n / c_param))
        
        return Th
    
    @property
    def Tv_g(self):
        return self._Tv_g
    
    @property
    def Tv_d(self):
        return self._Tv_d
    
    def cardan(self, a, b, L0, cable_temperature):
        pass
    
    @property
    def parameter(self):
        return self._parameter

    def find_parameter(self, parameter, a, b, L0, cable_temperature):
        pass
    
    def update_span(self):
        """transmet_portee"""
        # warning for dev : we dont use the first element of span vectors for the moment
        a = self.nodes._x + self.nodes.dx - np.roll(self.nodes._x + self.nodes.dx, 1)
        b = self.nodes._z + self.nodes.dz - np.roll(self.nodes._z + self.nodes.dz, 1)
        self.a = a[1:]
        self.b = b[1:]
        
    def z_from_x_2ddl(self):
        # Assuming this is a placeholder for the actual implementation
        # warning here : this is not the same as the function update_span (i+1) - (i-1) instead of (i) - (i-1)
        a = np.roll(self.nodes.x + self.nodes.dx, -1) - np.roll(self.nodes.x + self.nodes.dx, 1)
        b = np.roll(self.nodes.z + self.nodes.dz, -1) - np.roll(self.nodes.z + self.nodes.dz, 1)
        
        
        z = self.nodes.z
        x_m = f.x_m(a, b, self.parameter)
        zdz_im1 = np.roll(self.nodes.z + self.nodes.dz, 1)
        xdx_im1 = np.roll(self.nodes.x + self.nodes.dx,1)
        xdx_i = self.nodes.x + self.nodes.dx
        
        z_i = zdz_im1 + f.z(xdx_i - xdx_im1, self.parameter, x_m) - f.z(0*xdx_i, self.parameter, x_m)

        return np.where(self.nodes.ntype == 1, z_i, z)
    

    
    def vector_force(self):
        self.nodes.compute_forces(self.Th, self.Tv_d, self.Tv_g, self.parameter)
        out = np.vstack((self.nodes.Fx, self.nodes.Fz, self.nodes.My))
        return np.reshape(out, -1, order = 'F')
    
    def _delta(self, dz_se_only):
        self.nodes.dz[0] = dz_se_only[0]
        self.nodes.dz[-1] = dz_se_only[-1]
        # self.update_span()
        self.z_from_x_2ddl()
        force_vector = self.vector_force()
        
        return force_vector
    
    def compute_balance(self):
        

        
        def norm_delta(dz_se_only):
            return np.linalg.norm(self._delta(dz_se_only))
            
        dz = optimize.newton(norm_delta, np.array([.0001, .0001]))
        
        
    def __repr__(self):
        
        data = {
            'parameter': self.parameter,
            'cable_temperature': self.cable_temperature,
            'Th': self.Th,
            'Tv_d': self.Tv_d,
            'Tv_g': self.Tv_g,
        }
        out = pd.DataFrame(data)
        
        return str(out)

    
    def __str__(self):
        return self.__repr__()
            

        
 
    
    
class Nodes:
    def __init__(
        self,
        # num: np.ndarray,
        ntype: np.ndarray,
        L_chain: np.ndarray,
        weight_chain: np.ndarray,
        x: np.ndarray,
        z: np.ndarray,
        # dx: np.ndarray,
        # dz: np.ndarray,
        load: np.ndarray
    ):
        self.num = np.arange(len(ntype))
        self.ntype = ntype
        self.L_chain = L_chain
        self.weight_chain = weight_chain
        self._x = x
        self._z = z
        self.dx = np.zeros_like(x)
        self.dz = np.zeros_like(z)
        self.load = load
        # self.dz = dz
        self.init_L()
        # self.compute()
    
    def __len__(self):
        return len(self.num)
    
    @property
    def x(self):
        return self._x + self.L_anchor_chain
    
    @property
    def z(self):
        return self._z
    
    def init_L(self):
        self.L_anchor_chain = np.zeros_like(self._x)
        self.L_anchor_chain[0] = self.L_chain[0]
        self.L_anchor_chain[-1] = -self.L_chain[-1]

    
    def compute_dx_dz(self):
        L = self.L_chain
        dz2 = L - (L**2-self.dx**2)**.5
        dx1s = -L + (L**2 - self.dz **2)**.5
        dx1e = -(L**2 - self.dz**2)**.5 + L
        
        self.dz = np.where(self.ntype == 2, dz2, self.dz)
        self.dx[0] = dx1s[0]
        self.dx[-1] = dx1e[-1]
        
    
    
    def compute_forces(self, Th, Tv_d, Tv_g, parameter):
        # Placeholder for force computation logic
        # case 1: ntype == 1
        self.compute_dx_dz()
        
        Th_i = np.concat((np.array([0]), Th))
        Th_ip1 = np.concat((Th, np.array([0])))
        Tv_d_i = np.concat((np.array([0]), Tv_d))
        Tv_g_ip1 = np.concat((Tv_g, np.array([0])))
        
        Fx = -Th_i + Th_ip1 # -Th_i + Th_i
        Fz = -Tv_d_i + Tv_g_ip1 + self.weight_chain/2 + self.load # -Tvd_i + Tvg_i
        
        def compute_forces_2ddl():
            Fx = np.roll(Th, -1) - Th # Th_i+1 - Th_i
            Fz = Tv_d + np.roll(Tv_g, 1) + self.load # >> n_charge
            My = 0
            return Fx, Fz, My
        
        L = self.L_chain
        # case 2: ntype == 2
        def compute_forces_1ddlvertical(): 
            
            Fx = np.roll(Th, -1) - Th
            Fz = -Tv_d + np.roll(Tv_g, -1) + self.weight_chain/2 # -Tvd_i + Tvg_i+1 + parameter/2
            My = Fz * self.dx - Fx *(L - self.dz)
            return Fx, Fz, My
        
        # case 3: ntype == 3 + starting
        
        def compute_forces_1ddlhorizontal_starting():
            # case 2: ntype == 2

            Fx = np.roll(Th, -1) # Th_i+1
            Fz = np.roll(Tv_g, 0) + self.weight_chain/2 # Tvg_i+1 + parameter/2
            My = Fz *(L + self.dx)+Fx*self.dz 
            return Fx, Fz, My
        
        # case 3: ntype == 3 + ending
        def compute_forces_1ddlhorizontal_ending():
            
            Fx = -Th # -Th_i
            Fz = Tv_d + self.weight_chain/2
            My = Fz * (L - self.dx) - Fx * self.dz
            return Fx, Fz, My
        
        
        
        Fx2, Fz2, My2 = compute_forces_2ddl()
        Fx1v, Fz1v, My1v = compute_forces_1ddlvertical()
        Fx1hs, Fz1hs, My1hs = compute_forces_1ddlhorizontal_starting()
        Fx1he, Fz1he, My1he = compute_forces_1ddlhorizontal_ending()
        
        Fx = Fx2
        Fz = Fz2
        My = My2
        
        Fx = np.where(self.ntype == 2, Fx1v, Fx)
        Fx[0] = Fx1hs[0]
        Fx[-1] = Fx1he[-1]
        
        Fz = np.where(self.ntype == 2, Fz1v, Fz)
        Fz[0] = Fz1hs[0]
        Fz[-1] = Fz1he[-1]
        
        My = np.where(self.ntype == 2, My1v, My)
        My[0] = My1hs[0]
        My[-1] = My1he[-1]
        # Fz = np.where(self.nodes.ntype == 2, compute_for
        
        self.Fx = Fx
        self.Fz = Fz
        self.My = My
        
        return Fx, Fz, My # veteur combiné des forces et moments
    
    # self compute_forces(self):
    #     Fx = 
        
    #     np.where(self.ntype == 2, self.Fx(), 0)
    
    def __repr__(self):
        
        data = {
            'num': self.num,
            'ntype': self.ntype,
            'L_chain': self.L_chain,
            'weight_chain': self.weight_chain,
            'x': self.x,
            'z': self.z,
            'dx': self.dx,
            'dz': self.dz,
            'load': self.load,
            'Fx': self.Fx,
            'Fz': self.Fz,
            'My': self.My
        }
        out = pd.DataFrame(data)
        
        return str(out)

    
    def __str__(self):
        return self.__repr__()


class SolverBalance:
    
    eps = 1e-4
    max_iter = 250
    
    
    def __init__(self, section):
        self.section = section
        # self.f = self._d
    
    def solve(self, x0=np.array([0, 0,]) ):
        eps = self.eps
        
        for i in range(self.max_iter):

            
            force_vector = self.section._delta(x0)
                        
            d_force_vector = self.section._delta(x0+eps)
            
            delta = force_vector / (d_force_vector - force_vector)
            
            x0 = (x0 - eps) - delta * eps
        
        
        
        # def _delta(dz_se_only):
        #     self.nodes.dz[0] = dz_se_only[0]
        #     self.nodes.dz[-1] = dz_se_only[-1]
        #     # self.update_span()
            
        #     force_vector = self.vector_force()
            
        #     return np.linalg.norm(force_vector)