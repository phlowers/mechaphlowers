from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

import mechaphlowers.core.models.balance.functions as f
import mechaphlowers.core.numeric.numeric as optimize


class NodeType(Enum):
    DDL_2 = 1
    DDL_1V = 2
    DDL_1H = 3

@dataclass
class Cable:
    section: np.ndarray
    lineic_weight: np.ndarray
    dilation_coefficient: np.ndarray
    young_modulus: np.ndarray


@dataclass
class Span:
    a: np.ndarray
    b: np.ndarray
    length_0: np.ndarray
    cable_temperature: np.ndarray
    reglage: np.ndarray
    nodes: Nodes
    finition: bool = False
    _parameter: np.ndarray
    cable: Cable
    
    def Th(self):
        if not self.reglage:
            c_param = self.cardan(self.a, self.b, self.length_0, self.cable_temperature)
            if self.finition:
                c_param = self.find_parameter(self.parameter, self.a, self.b, self.length_0, self.cable_temperature)
        else:
            c_param = self._parameter
            
        Th = c_param * self.cable.lineic_weight
        
        x_m = f.x_m(self.a, self.b, c_param)
        x_n = f.x_n(self.a, self.b, c_param)
        
        self._Tv_g = -Th * (np.sinh(x_m / c_param)) # moins ?
        self._Tv_d = Th * (np.sinh(x_n / c_param))
    
    def Tv_g(self):
        return self._Tv_g
    
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
        self.a = self.nodes.x + self.nodes.dx - np.roll(self.nodes.x + self.nodes.dx, 1)
        self.b = self.nodes.z + self.nodes.dz + np.roll(self.nodes.z + self.nodes.dz, 1)
        
    def z_from_x_2ddl(self):
        # Assuming this is a placeholder for the actual implementation
        x_m = f.x_m(self.a, self.b, self.parameter)
        new_x = -np.roll(self.nodes.x + self.nodes.dx, 1) + self.nodes.x + self.nodes.dx
        new_z = np.roll(self.nodes.z + self.nodes.dz, 1) + f.z(new_x, self.parameter, x_m) - f.z(0, self.parameter, x_m)
        return np.where(self.nodes.ntype == 2, new_z, np.zeros_like(new_z))
    

    
    def vector_force(self):
        self.nodes.compute_forces(self.Th, self.Tv_d, self.Tv_g, self.parameter)
        out = np.vstack((self.nodes.Fx, self.nodes.Fz, self.nodes.My))
        return np.reshape(out, -1, order = 'F')
    
    def minimize_function(self):
        
        def _delta(dz_se_only):
            self.nodes.dz[0] = dz_se_only[0]
            self.nodes.dz[-1] = dz_se_only[-1]
            self.update_span()
            force_vector = self.vector_force()
            
            return np.linalg.norm(force_vector)
            
        dz = optimize.newton(_delta, np.array([.0001, .0001]))
            

        
 
    
    
@dataclass
class Nodes:
    num: np.ndarray
    ntype: np.ndarray
    L_chain: np.ndarray
    weight_chain: np.ndarray
    _x: np.ndarray
    z: np.ndarray
    dx: np.ndarray
    dz: np.ndarray
    load: np.ndarray
    
    
    @property
    def x(self):
        return self._x + self.L_anchor_chain
    
    def init_L(self):
        self.L_anchor_chain = np.zeros_like(self.x)
        self.L_anchor_chain[0] = self.L_chain[0]
        self.L_anchor_chain[-1] = -self.L_chain[-1]
        
    
    def compute(self):
        self.compute_forces()
        self.compute_moments()
        
    # def Fx(self):
    #     pass
    
    # def Fz(self):
    #     pass
    
    # def My(self):
    #     pass
    
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
        def compute_forces_2ddl():
            Fx = -Th + np.roll(Th, -1)
            Fz = -Tv_d + np.roll(Tv_g, -1) + self.load # >> n_charge
            My = 0
            return Fx, Fz, My
        
        L = self.L_chain
        # case 2: ntype == 2
        def compute_forces_1ddlvertical(): 
            
            Fx = -Th + np.roll(Th, -1)
            Fz = -Tv_d + np.roll(Tv_g, -1) + parameter # >>> n_p/2 ??
            My = Fz * self.dx - Fx *(L - self.dz)
            return Fx, Fz, My
        
        # case 3: ntype == 3 + starting
        
        def compute_forces_1ddlhorizontal_starting():
            # case 2: ntype == 2

            Fx = np.roll(Th, -1)
            Fz = np.roll(Tv_g, -1) + parameter/2
            My = Fz *(L + self.dx)+Fx*self.dz
            return Fx, Fz, My
        
        # case 3: ntype == 3 + ending
        def compute_forces_1ddlhorizontal_ending():
            
            Fx = Th
            Fz = Tv_d + parameter/2
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
    
    

    

