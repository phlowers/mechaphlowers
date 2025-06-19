from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mechaphlowers.core.models.balance.functions as f


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
    nodes: Node
    
    def Th(self):
        if not self.reglage:
            c_param = self.cardan(self.a, self.b, self.length_0, self.cable_temperature)
            if self.finition:
                c_param = self.find_parameter(self.parameter, self.a, self.b, self.length_0, self.cable_temperature)
        else:
            c_param = self._parameter
            
        Th = c_param *cable.lineic_weight
        
        x_m = -a/2 + c_param * np.arcsinh(b / (2 * c_param * np.sinh(a / (2 * c_param))))
        x_n = x_m + a
        
        Tvg = -Th * (np.sinh(2*x_m / c_param))
        Tvd = Th * (np.sinh(2*x_n / c_param))
    
    def Tv_g(self):
        pass
    
    def Tv_d(self):
        pass
    
    def cardan(self, a, b, L0, cable_temperature):
        pass
    
    @property
    def parameter(self):
        pass

    def find_parameter(self, parameter, a, b, L0, cable_temperature):
        pass
    
    def compute(self):
        self.a = self.nodes.x + self.nodes.dx - np.roll(self.nodes.x + self.nodes.dx, 1)
        self.b = self.nodes.z + self.nodes.dz + np.roll(self.nodes.z + self.nodes.dz, 1)
        
    def z_from_x(self, x):
        # Assuming this is a placeholder for the actual implementation
        x_m = f.x_m(self.a, self.b, self.parameter)
        new_x = np.roll(self.nodes.x + self.nodes.dx, 1) - self.nodes.x - self.nodes.dx
        new_z = np.roll(self.nodes.z + self.nodes.dz, 1) + f.z(new_x, self.parameter, x_m) - f.z(0, self.parameter, x_m)
        return np.where(self.nodes.ntype == 2, new_z, np.zeros_like(new_z))
    
    def compute_forces(self):
        # Placeholder for force computation logic
        # case 1: ntype == 1
        Fx = -self.Th + np.roll(self.Th, -1)
        Fz = -self.Tv_d + np.roll(self.Tv_g, -1) + self.load # >> n_charge
        My = 0
        
        # case 2: ntype == 2
        dz = L - (L**2-dx**2)**.5
        Fx = -self.Th + np.roll(self.Th, -1)
        Fz = -self.Tv_d + np.roll(self.Tv_g, -1) + self.load # >>> n_p/2 ??
        My = Fz * dx - Fx *(L - dz)
        
        # case 3: ntype == 3 + starting
        dx = -L + (L**2 - dz **2)**.5
        Fx = np.roll(self.Th, -1)
        Fz = np.roll(self.Tv_g, -1) + n_p/2
        My = Fz *(L+dx)+Fx*dz
        
        # case 3: ntype == 3 + ending
        dx = -(L**2 - dz**2)**.5 + L
        Fx = self.Th
        Fz = self.Tv_d + n_p/2
        My = Fz * (L - dx) - Fx * dz
        
        return Fx, Fz, My # veteur combiné des forces et moments
    
    
@dataclass
class Node:
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
        
    def Fx(self):
        pass
    
    def Fz(self):
        pass
    
    def My(self):
        pass
    
    # self compute_forces(self):
    #     Fx = 
        
    #     np.where(self.ntype == 2, self.Fx(), 0)

