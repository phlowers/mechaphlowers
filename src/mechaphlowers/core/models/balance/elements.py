from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from mechaphlowers.core.models.balance import numeric
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
        # self.init_Lref_mode = True
        # self.update_span()
        # self.nodes.no_load = True
        # self.nodes.compute_forces(self.Th, self.Tv_d, self.Tv_g, self.parameter)
        self.nodes.no_load = True
        self.init_Lref_mode = False
        self.update_span()
        self.nodes.compute_forces(self.Th, self.Tv_d, self.Tv_g, self.parameter)
        self.nodes.no_load = False
    


    
    @property
    def Th(self):
        self.update_tensions()
        return self._Th
        
    def update_tensions(self):
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
        self._Th = Th
        
        return Th
    
    @property
    def Tv_g(self):
        self.update_tensions()
        return self._Tv_g
    
    @property
    def Tv_d(self):
        self.update_tensions()
        return self._Tv_d
    
    def cardan(self, a, b, L0, cable_temperature):
        circle_chord = (a**2 + b**2)**0.5
        
        factor = self.cable.lineic_weight / self.cable.young_modulus /self.cable.section
        
        p3 = factor * L0
        p2 = L0 - circle_chord + self.cable.dilation_coefficient * cable_temperature * L0
        p1 = 0
        p0 = -a**4 / 24 / circle_chord
        
        # cubic_roots(np.array([[-1, -3, 2, 3], [-10, -3, 2, 7]]))
        # we have to do p3 * x**3 + p2 * x**2 + p1 * x + p0 = 0
        # p = p3 | p2 | p1 | p0
        # then 
        
        roots = numeric.cubic_roots(p)
        return roots
    
    @property
    def parameter(self):
        return self._parameter

    def find_parameter(self, parameter, a, b, L0, cable_temperature):
        pass
     
    
    def update_span(self):
        """transmet_portee"""
        # warning for dev : we dont use the first element of span vectors for the moment
        if self.init_Lref_mode is True:
            x = self.nodes._x
            z = self.nodes._z
        else:
            x = self.nodes.x
            z = self.nodes.z
        a = x + self.nodes.dx - np.roll(x + self.nodes.dx, 1)
        b = z + self.nodes.dz - np.roll(z + self.nodes.dz, 1)
        self.a = a[1:]
        self.b = b[1:]
        
    def z_from_x_2ddl(self, inplace=True):
        # Assuming this is a placeholder for the actual implementation
        # warning here : this is not the same as the function update_span (i+1) - (i-1) instead of (i) - (i-1)
        a = np.roll(self.nodes.x + self.nodes.dx, -1) - np.roll(self.nodes.x + self.nodes.dx, 1)
        b = np.roll(self.nodes.z + self.nodes.dz, -1) - np.roll(self.nodes.z + self.nodes.dz, 1)
        # a = a[:-1]
        # b = b[:-1]
        
        z = self.nodes.z
        parameter_np1 = np.hstack((self.parameter, self.parameter[-1]))
        x_m = f.x_m(a, b, parameter_np1)
        zdz_im1 = np.roll(self.nodes.z + self.nodes.dz, 1)
        xdx_im1 = np.roll(self.nodes.x + self.nodes.dx,1)
        xdx_i = self.nodes.x + self.nodes.dx
        
        z_i = zdz_im1 + f.z(xdx_i - xdx_im1, parameter_np1, x_m) - f.z(0*xdx_i, parameter_np1, x_m)
        if inplace is True:
            self.nodes.z = np.where(self.nodes.ntype == 1, z_i, z)
        return np.where(self.nodes.ntype == 1, z_i, z)
    

    
    def vector_force(self):
        self.nodes.compute_forces(self.Th, self.Tv_d, self.Tv_g, self.parameter)
        # out = np.vstack((self.nodes.Fx, self.nodes.Fz, self.nodes.My))
        
        # manual fix to go on
        # TODO: vectorize
        out = []
        for i in range(0, len(self.nodes.Fx)):
            if self.nodes.ntype[i] == 1:
                out.append(self.nodes.Fx[i])
                out.append(self.nodes.Fz[i])
            if self.nodes.ntype[i] == 2 or self.nodes.ntype[i] == 3:
                out.append(self.nodes.My[i])
                
        out = np.array(out)        
        return out # np.reshape(out, -1, order = 'F')
    
    def _delta(self, dz_se_only):
        self.nodes.dz[0] = dz_se_only[0]
        self.nodes.dz[-1] = dz_se_only[-1]
        # self.update_span()
        # self.z_from_x_2ddl()
        self.update_span()
        self.update_tensions()
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
            


class RealSpan:
    def __init__(self, span: Span):
        self.span = span
    
    def real_span(self):
# a = noeud(i + 1).x + noeud(i + 1).dx - noeud(i - 1).x - noeud(i - 1).dx
        a = np.roll(self.span.nodes.x + self.span.nodes.dx, -1) - np.roll(self.span.nodes.x + self.span.nodes.dx, 1)
        b = np.roll(self.span.nodes.z + self.span.nodes.dz, -1) - np.roll(self.span.nodes.z + self.span.nodes.dz, 1)
# b = noeud(i + 1).z + noeud(i + 1).dz - noeud(i - 1).z - noeud(i - 1).dz

# lon1 = longueur(param_reglage, a + pt_x_m(a, b, param_reglage), pt_x_m(a, b, param_reglage))
        lon1 = f.L(self.span.parameter, a+f.x_m(a, b, self.span.parameter), f.x_m(a, b, self.span.parameter))
# Tm1 = T_moy(param_reglage, a + pt_x_m(a, b, param_reglage), pt_x_m(a, b, param_reglage), lon1)
        Tm1 = f.T_moy(self.span.parameter, a+f.x_m(a, b, self.span.parameter), f.x_m(a, b, self.span.parameter))
# lon1 = lon1 / (1 + Temp * cable.coef_dilat + Tm1 / cable.mod_young / cable.Section)
        lon1 = lon1 / (1 + self.span.cable.dilation_coefficient * self.span.cable_temperature + Tm1 / self.span.cable.young_modulus  / self.span.cable.section)

# pos_charge = noeud(i).x - noeud(i - 1).x - noeud(i - 1).dx
        pos_charge = self.span.nodes.x - np.roll(self.span.nodes.x,1) - np.roll(self.span.nodes.dx,1)

# lon2 = longueur(param_reglage, pos_charge + pt_x_m(a, b, param_reglage), pt_x_m(a, b, param_reglage))
        lon2 = f.L(self.span.parameter, pos_charge+f.x_m(a, b, self.span.parameter), f.x_m(a, b, self.span.parameter))
# Tm2 = T_moy(param_reglage, pos_charge + pt_x_m(a, b, param_reglage), pt_x_m(a, b, param_reglage), lon2)
        Tm2 = f.T_moy(self.span.parameter, pos_charge+f.x_m(a, b, self.span.parameter), f.x_m(a, b, self.span.parameter))
# lon2 = lon2 / (1 + Temp * cable.coef_dilat + Tm2 / cable.mod_young / cable.Section)
        lon2 = lon2 / (1 + self.span.cable.dilation_coefficient * self.span.cable_temperature + Tm2 / self.span.cable.young_modulus  / self.span.cable.section)
# portee(i).L0 = lon2
        L0_i = lon2
# portee(i + 1).L0 = lon1 - lon2
        L0_ip1 = lon1 - lon2
        self.L0 = np.where(self.span.nodes.ntype==1, L0_i, L0_ip1)

        
        
        
 
    
    
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
        self.dx = np.zeros_like(x, dtype=np.float64)
        self.dz = np.zeros_like(z, dtype=np.float64)
        self._load = load
        self.no_load = False
        # self.dz = dz
        self.init_L()
        # self.compute()
    
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
        return self._x + self.L_anchor_chain
    
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, value):
        self._z = value
    
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
        
        assert True
        
    
    
    def compute_forces(self, Th, Tv_d, Tv_g, parameter):
        # Placeholder for force computation logic
        # case 1: ntype == 1
        self.compute_dx_dz()
        
        Th_i = np.concat((np.array([0]), Th))
        Th_ip1 = np.concat((Th, np.array([0])))
        Tv_d_i = np.concat((np.array([0]), Tv_d))
        Tv_g_ip1 = np.concat((Tv_g, np.array([0])))
        
        Fx = -Th_i + Th_ip1 # -Th_i + Th_i
        Fz = Tv_d_i + Tv_g_ip1 + self.weight_chain/2 + self.load # -Tvd_i + Tvg_i
        
        # base_build = np.concat(np.array([0,1]*int((len(self)-2-1)/2)), np.array([0]))
        base_build = np.array([0,1]*int((len(self)-2-1)/2))
        base_build = np.concat((base_build, np.array([0])))
        
        
        force_3d = np.vstack((Fx, np.zeros_like(Fx), Fz)).T
        moment_length_3d = np.vstack((base_build*self.dx[1:-1], np.zeros_like(base_build), base_build*(self.L_chain[1:-1]-self.dz[1:-1]))).T
        moment_length_3d = np.vstack((
            np.array([self.L_chain[0]+self.dx[0], 0, -self.dz[0]]).T,
            moment_length_3d,
            np.array([self.L_chain[0]-self.dx[-1], 0, self.dz[-1]]).T,
            
        ))
        
        M = np.cross(force_3d, moment_length_3d)
        My = M[:,1]
        
        self.Fx = Fx
        self.Fz = Fz
        self.My = My
        
        return Fx, Fz, My # veteur combiné des forces et moments
    
    # self compute_forces(self):
    #     Fx = 
        
    #     np.where(self.ntype == 2, self.Fx(), 0)

    def debug(self):
        
        data = {
            'z': self.dz,
            'x': self.dx,
            'dz': self.z,
            'Fx': self.Fx,
            'Fz': self.Fz,
            'My': self.My
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
        self.section: Span = section
        # self.f = self._d
    
    def solve(self, x0=np.array([0, 0,]) ):
        eps = self.eps
        
        for i in range(self.max_iter):

            self.section.z_from_x_2ddl()         
            force_vector = self.section._delta(x0)
                        
            d_force_vector = self.section._delta(x0+eps)
            
            delta = force_vector[[0,-1]] / (d_force_vector[[0,-1]] - force_vector[[0,-1]])
            
            x0 = (x0 - eps) - delta * eps
            
            if np.linalg.norm(force_vector) < 1.:
                break
        
        return self.section.dz
            
            
        
        
        
        # def _delta(dz_se_only):
        #     self.nodes.dz[0] = dz_se_only[0]
        #     self.nodes.dz[-1] = dz_se_only[-1]
        #     # self.update_span()
            
        #     force_vector = self.vector_force()
            
        #     return np.linalg.norm(force_vector)
        
        
        
        
        