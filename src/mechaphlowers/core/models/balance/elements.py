from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import numpy as np
import pandas as pd

from mechaphlowers.core.models.balance import numeric
import mechaphlowers.core.models.balance.functions as f
import mechaphlowers.core.numeric.numeric as optimize

from mechaphlowers.utils import ppnp


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
        # TODO: during L_ref computation, perhaps set cable_temperature = 0
        # TODO: temperature here is tuning temperature / only in the real span part
        # TODO: there is another temperature : change state
        
        self.update_span()
        self.nodes.compute_forces(self.Th, self.Tv_d, self.Tv_g, self.parameter)
        self.nodes.no_load = False
        
        SolverBalance(self).adjusting_Lref()
        
        self._L_ref = RealSpan(self).real_span()
        
    

    @property
    def L_ref(self):
        return self._L_ref
    

    def update_length(self):
        RealSpan(self).real_span()
    
    @property
    def Th(self):
        self.update_tensions()
        return self._Th
    
    # TODO: méthode sur appelée, mettre en lru_cache ?   
    def update_tensions(self):
        if not self.reglage:
            c_param = self.cardan(self.a, self.b, self.L_ref, self.cable_temperature)
            self._parameter = c_param
            if self.finition:
                c_param = self.find_parameter(self.parameter, self.a, self.b, self.L_ref, self.cable_temperature)
                self._parameter = c_param
        else:
            c_param = self._parameter
            
        Th = c_param * self.cable.lineic_weight * np.ones(len(self.nodes)-1)
        
        x_m = f.x_m(self.a, self.b, c_param)
        x_n = f.x_n(self.a, self.b, c_param)
        
        self._Tv_g = Th * (np.sinh(x_m / c_param)) 
        self._Tv_d = -Th * (np.sinh(x_n / c_param))
        self._Th = Th
        
        return Th
    
    @property
    def Tv_g(self):
        # self.update_tensions()
        return self._Tv_g
    
    @property
    def Tv_d(self):
        # self.update_tensions()
        return self._Tv_d
    
    def get_approximative_parameter(self):
        self._parameter = self.cardan(self.a, self.b, self.L_ref, self.cable_temperature)
        
    
    def cardan(self, a, b, L0, cable_temperature):
        circle_chord = (a**2 + b**2)**0.5
        
        factor = self.cable.lineic_weight / self.cable.young_modulus /self.cable.section
        
        p3 = factor * L0
        p2 = L0 - circle_chord + self.cable.dilation_coefficient * cable_temperature * L0
        p1 = 0 *L0
        p0 = -a**4 / 24 / circle_chord
        
        # cubic_roots(np.array([[-1, -3, 2, 3], [-10, -3, 2, 7]]))
        # we have to do p3 * x**3 + p2 * x**2 + p1 * x + p0 = 0
        # p = p3 | p2 | p1 | p0
        # then 
        p = np.vstack((p3, p2, p1, p0)).T
        roots = numeric.cubic_roots(p)
        return roots.real
    
    @property
    def parameter(self):
        return self._parameter



# param = param_init

# compteur = 0

# Do ' méthode Newton

#     compteur = compteur + 1 ' au cas où ça plante...
    
#     mem = param

#     x_m = -port / 2 + param * WorksheetFunction.Asinh(deniv / (2 * param * WorksheetFunction.Sinh(port / 2 / param)))
#     x_n = x_m + port
    
#     lon = param * WorksheetFunction.Sinh(x_n / param) - param * WorksheetFunction.Sinh(x_m / param)
    
#     Tmoy = param * cable.pds_lin * (port + (WorksheetFunction.Sinh(2 * x_n / param) - WorksheetFunction.Sinh(2 * x_m / param)) * param / 2) / lon / 2
    
#     delta1 = ((lon - L0) / L0 - (cable.coef_dilat * T + Tmoy / (cable.mod_young) / cable.Section))
    
#     param = param + 1
    
#     x_m = -port / 2 + param * WorksheetFunction.Asinh(deniv / (2 * param * WorksheetFunction.Sinh(port / 2 / param)))
#     x_n = x_m + port
    
#     lon = param * WorksheetFunction.Sinh(x_n / param) - param * WorksheetFunction.Sinh(x_m / param)
    
#     Tmoy = param * cable.pds_lin * (port + (WorksheetFunction.Sinh(2 * x_n / param) - WorksheetFunction.Sinh(2 * x_m / param)) * param / 2) / lon / 2
    
#     delta2 = ((lon - L0) / L0 - (cable.coef_dilat * T + Tmoy / (cable.mod_young) / cable.Section))
    
#     param = (param - 1) - delta1 / (delta2 - delta1)

# Loop Until Abs(mem - param) < 0.001 Or compteur > 100

# trouve = param

# If compteur = 101 Then
#     msgbox "plante Newton"
#     Stop
# End If



    # @f.np_cache(maxsize=256)
    def find_parameter(self, parameter, a, b, L0, cable_temperature):
        param = parameter
        
        n_iter = 50
        
        for i in range(n_iter):
            x_m = f.x_m(a,b,param)
            x_n = f.x_n(a,b,param)
            lon = f.L(param, x_n, x_m)
            Tm1 = f.T_moy(
                p=param, 
                L=lon,
                x_n=a+f.x_m(a, b, param), 
                x_m=f.x_m(a, b, param),
                lineic_weight=self.cable.lineic_weight,
                )
            
            delta1 = ((lon - self.L_ref) / self.L_ref - (self.cable.dilation_coefficient * self.cable_temperature + Tm1 / (self.cable.young_modulus) / self.cable.section))
            
            mem = param
            param = param + 1
            
            x_m = f.x_m(a,b,param)
            x_n = f.x_n(a,b,param)
            lon = f.L(param, x_n, x_m)
            Tm1 = f.T_moy(
                p=param, 
                L=lon,
                x_n=a+f.x_m(a, b, param), 
                x_m=f.x_m(a, b, param),
                lineic_weight=self.cable.lineic_weight,
                )
            
            delta2 = ((lon - self.L_ref) / self.L_ref - (self.cable.dilation_coefficient * self.cable_temperature + Tm1 / (self.cable.young_modulus) / self.cable.section))
            
            param = (param - 1) - delta1 / (delta2 - delta1)
            # print(f"delta param {str(mem - param)}")
#TODO: tune the threshold
            if np.linalg.norm(mem - param) < .1*param.size:
                print("--end--")
                print(f"{i=}, {param=}")
                break
            if i == n_iter:
                print("max iter reached")
        # print(f"delta param {str(np.linalg.norm(mem - param))}")
        # print("param = ", param)
        
        return param    
         
    
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
    

    
    def vector_force(self, update_dx_dz = True):
        self.nodes.compute_forces(self.Th, self.Tv_d, self.Tv_g, self.parameter, update_dx_dz=update_dx_dz)
        # out = np.vstack((self.nodes.Fx, self.nodes.Fz, self.nodes.My))
        
        # manual fix to go on
        # TODO: vectorize
        out = self.build_vector_force()        
        return out 

    def build_vector_force(self):
        out = []
        for i in range(0, len(self.nodes.Fx)):
            if self.nodes.ntype[i] == 1:
                out.append(self.nodes.Fx[i])
                out.append(self.nodes.Fz[i])
            if self.nodes.ntype[i] == 2 or self.nodes.ntype[i] == 3:
                out.append(self.nodes.My[i])
                
        out = np.array(out)
        return out# np.reshape(out, -1, order = 'F')
    
    def _delta_init_L(self, dz_se_only):
        self.nodes.dz[0] = dz_se_only[0]
        self.nodes.dz[-1] = dz_se_only[-1]
        # self.update_span()
        # self.z_from_x_2ddl()
        self.update_span()
        self.update_tensions()
        force_vector = self.vector_force()
        
        return force_vector
    
    def _delta_dz(self, dz):
        
        f1 = self.vector_force(False)
        self.nodes.dz += dz
        # self.update_span()
        # self.z_from_x_2ddl()
        
        self.nodes.compute_dx_dz()
        self.update_span() # transmet_portee: update a and b
        

        # self.get_approximative_parameter()
        self.update_tensions() # Th : cardan for parameter then compute Th, Tvd, Tvg

        force_vector = self.vector_force(update_dx_dz=False)
        
        self.nodes.dz -= dz
        # TODO: check why the state f2 is not the same as f1
        self.nodes.compute_dx_dz()
        self.update_span()

        self.update_tensions()
        f2 = self.vector_force(update_dx_dz=False)
        
        return force_vector
    
    def _delta_dx(self, dx):
        self.nodes.dx += dx
        # self.update_span()
        # self.z_from_x_2ddl()
        self.update_span()
        self.nodes.compute_dx_dz()
        self.update_tensions()
        
        force_vector = self.vector_force(update_dx_dz=False)
        
        
        self.nodes.dx -= dx
        # self.nodes.compute_dx_dz()
        # self.update_span()
        # self.update_tensions()
        
        return force_vector
    
    def compute_balance(self):
        

        
        def norm_delta(dz_se_only):
            return np.linalg.norm(self._delta_init_L(dz_se_only))
            
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
        
        # TODO: this part take the hypothesis that there is one of two node of ntype 1.
        # we should change this to be more general
        
        a = np.roll(self.span.nodes.x + self.span.nodes.dx, -1) - np.roll(self.span.nodes.x + self.span.nodes.dx, 1)
        b = np.roll(self.span.nodes.z + self.span.nodes.dz, -1) - np.roll(self.span.nodes.z + self.span.nodes.dz, 1)

        parameter_np1 = np.hstack((self.span.parameter, self.span.parameter[-1]))

        lon1 = f.L(parameter_np1, a+f.x_m(a, b, parameter_np1), f.x_m(a, b, parameter_np1))

        Tm1 = f.T_moy(
            p=parameter_np1, 
            L=lon1,
            x_n=a+f.x_m(a, b, parameter_np1), 
            x_m=f.x_m(a, b, parameter_np1),
            lineic_weight=self.span.cable.lineic_weight,
            )

        lon1 = lon1 / (1 + self.span.cable.dilation_coefficient * self.span.cable_temperature + Tm1 / self.span.cable.young_modulus  / self.span.cable.section)

        pos_charge = self.span.nodes.x - np.roll(self.span.nodes.x,1) - np.roll(self.span.nodes.dx,1)

        lon2 = f.L(parameter_np1, pos_charge+f.x_m(a, b, parameter_np1), f.x_m(a, b, parameter_np1))

        Tm2 = f.T_moy(
            p=parameter_np1, 
            x_n=pos_charge+f.x_m(a, b, parameter_np1), 
            x_m=f.x_m(a, b, parameter_np1), 
            L=lon2,
            lineic_weight=self.span.cable.lineic_weight,
            )

        lon2 = lon2 / (1 + self.span.cable.dilation_coefficient * self.span.cable_temperature + Tm2 / self.span.cable.young_modulus  / self.span.cable.section)
        
        # we need np.array([lon2[1], lon1-lon2[1], lon2[3], ...])
        L_ref = np.reshape(
            np.vstack( # stacking the two array vertically and taking only 1/2 node
                (
                    lon2[1::2], 
                    lon1[1::2] - lon2[1::2])
                ),
            -1, order='F') # order='F' is for fortran order to flatten the array to get the good form
        
        return L_ref

        
        
        
 
    
    
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
        self.weight_chain = -weight_chain
        self._x = x
        self._z = z
        self.dx = np.zeros_like(x, dtype=np.float64)
        self.dz = np.zeros_like(z, dtype=np.float64)
        self._load = -load
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
        return self._x + self.x_anchor_chain
    
    @property
    def z(self):
        return self._z
    
    @z.setter
    def z(self, value):
        self._z = value
    
    def init_L(self):
        self.x_anchor_chain = np.zeros_like(self._x)
        self.x_anchor_chain[0] = self.L_chain[0]
        self.x_anchor_chain[-1] = -self.L_chain[-1]
        self.z_suspension_chain = np.zeros_like(self._x)
        self.z_suspension_chain[1:-1] = -self.L_chain[1:-1]

    
    def compute_dx_dz(self):

        L = self.L_chain
        
        # print(L**2 - self.dz **2)
        # print(L**2 - self.dx **2)
        
        dz2 = L - (L**2-self.dx**2)**.5
        dx1s = -L + (L**2 - self.dz **2)**.5
        dx1e = -(L**2 - self.dz**2)**.5 + L
        
        self.dz = np.where(self.ntype == 2, dz2, self.dz)
        self.dx[0] = dx1s[0]
        self.dx[-1] = dx1e[-1]
        
        assert True
        
    
    
    def compute_forces(self, Th, Tv_d, Tv_g, parameter, update_dx_dz=True):
        # Placeholder for force computation logic
        # case 1: ntype == 1
        if update_dx_dz:
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
        lever_arm = np.vstack((base_build*self.dx[1:-1], np.zeros_like(base_build), base_build*(self.z_suspension_chain[1:-1]+self.dz[1:-1]))).T
        lever_arm = np.vstack((
            np.array([self.x_anchor_chain[0]+self.dx[0], 0, self.dz[0]]).T,
            lever_arm,
            np.array([self.x_anchor_chain[-1]+self.dx[-1], 0, self.dz[-1]]).T,
            
        ))
        
        M = np.cross(lever_arm, force_3d)
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
    
    def adjusting_Lref(self, x0=np.array([0, 0,]) ):
        eps = self.eps
        force_vector_0 = self.section._delta_init_L(x0)
        
        for i in range(self.max_iter):

            self.section.z_from_x_2ddl() 
            
            force_vector = self.section._delta_init_L(x0)
                        
            d_force_vector = self.section._delta_init_L(x0+eps)
            
            delta = force_vector[[0,-1]] / (d_force_vector[[0,-1]] - force_vector[[0,-1]])
            
            x0 = (x0 - eps) - delta * eps
            
            if np.linalg.norm(force_vector) < 1.:
                break
        
        return self.section.nodes.dz
            
            
        
        
        
        # def _delta(dz_se_only):
        #     self.nodes.dz[0] = dz_se_only[0]
        #     self.nodes.dz[-1] = dz_se_only[-1]
        #     # self.update_span()
            
        #     force_vector = self.vector_force()
            
        #     return np.linalg.norm(force_vector)
        
        
def initialize_relaxation(nodes:Nodes, mask: np.ndarray, min_relaxation: float = .5):
    relaxation = (nodes.x - np.roll(nodes.x, 1)) / (np.roll(nodes.x, -1) - np.roll(nodes.x, 1))
    relaxation = relaxation[mask]
    relaxation = np.minimum(relaxation, 1-relaxation) # just absolute value ?
    
    if (relaxation < .01).any():
        raise ValueError("Load is too close to edge to solve")
    
    return 1-np.nanmin(np.append(relaxation, min_relaxation))

def solver_balance(section: Span, temperature=0, ):
    # get span temperature
    # input loads
    section.reglage = False
    section.nodes.no_load = False
    section.cable_temperature = temperature*np.ones_like(section.cable_temperature)
    
    #TODO:
    # section.trouve
    
    # cardan then update force
    section.get_approximative_parameter()
    
    section.update_tensions()
    section.update_span()
    
    # update ?
    
    # TODO: update cable temperature or new variable ?
    
    # compute relaxation
    relaxation = initialize_relaxation(section.nodes, mask=section.nodes.ntype==1)
    
    eps = .00001
    
    # only for values 
    finition = False
    
    x0 = np.zeros_like(section.nodes.x)
    
    # build jacobian matrix
    # section.z_from_x_2ddl()
    
    force_vector = section.vector_force()

    # masks for the jacobian
    # case 1 : two lines : one for dx one for dz
    
    n_iter = range(1,100)
    force_vector = section.vector_force()
    vector_eps = np.zeros_like(section.nodes.dx)
    pre_jacobian = np.zeros((len(force_vector),len(force_vector)))
    
    ## set relaxation to 0 for the moment
    # relaxation = 0.
    
    for compteur in n_iter:
        
        

        
        
        
        df_list = []
        
        for i in range(len(section.nodes.ntype)):
            vector_eps[i] += eps 
            # print(vector_eps)
            # print(force_vector)
            # print("eee"*10)
            
            if section.nodes.ntype[i] == 3:
                
                dz_d = section._delta_dz(vector_eps)
                vector_eps[i] -= eps 
                dF_dz = (dz_d - force_vector) / eps  
                df_list.append(dF_dz)
                # print(dz_d)
            elif section.nodes.ntype[i] == 2:
                dx_d = section._delta_dx(vector_eps)
                vector_eps[i] -= eps 
                dF_dx = (dx_d - force_vector) / eps 
                df_list.append(dF_dx)
                # print(dx_d)
            elif section.nodes.ntype[i] == 1:
                
                dx_d = section._delta_dx(vector_eps)
                dF_dx = (dx_d - force_vector) / eps 
                df_list.append(dF_dx)
                # print(dx_d)
                
                dz_d = section._delta_dz(vector_eps)
                vector_eps[i] -= eps 
                dF_dz = (dz_d - force_vector) / eps  
                df_list.append(dF_dz)
                # print(dz_d)
                   
            # dx_derivative, dz_derivative = compute_derivative(section, vector_eps)
            # vector_eps[i] -= eps    
        
            # dF_dx = (dx_derivative - force_vector) / eps
            # dF_dz = (dz_derivative - force_vector) / eps

            # if section.nodes.ntype[i] == 3:
            #     df_list.append(dF_dz)
            # elif section.nodes.ntype[i] == 2:
            #     df_list.append(dF_dx)
            # elif section.nodes.ntype[i] == 1:
            #     df_list.append(dF_dx)
            #     df_list.append(dF_dz)


        jacobian = np.array(df_list)
            


        # print("##"*10)
        # print(section.nodes.dx)
        # print(section.nodes.dz)
            
            
        
        # aa = np.array(section.nodes.ntype, copy=True)
        # insert_idx = np.where(aa==1)[0]
        # aa = np.insert(aa, insert_idx, 20*np.ones(len(insert_idx)))
        
        # insert_idx = np.where(aa==3)[0]
        # aa = np.insert(aa, insert_idx, 0*np.ones(len(insert_idx)))
        
        # insert_idx = np.where(aa==2)[0]
        # aa = np.insert(aa, insert_idx, 30*np.ones(len(insert_idx)))
        # aa[aa==2] = 0
        # aa[aa==20] = 2
        # aa[aa==30] = 3
        # print(aa!=0)
        # jacobian = pre_jacobian[aa!=0]
        
        mem = np.linalg.norm(force_vector)
        correction = np.linalg.inv(jacobian.T) @ force_vector

        
        i = 0
        for j in range(len(section.nodes.ntype)):
            
            #TODO: modify to delete update_tensions
            if section.nodes.ntype[j] == 3:
                section.nodes.dz[j] -= correction[i]*(1-relaxation**compteur)
                i += 1
                
            if section.nodes.ntype[j] == 2:
                section.nodes.dx[j] -= correction[i]*(1-relaxation**compteur)
                i += 1
            if section.nodes.ntype[j] == 1:
                section.nodes.dx[j] -= correction[i]*(1-relaxation**compteur)
                section.nodes.dz[j] -= correction[i+1]*(1-relaxation**compteur)
                i += 2
            section.nodes.compute_dx_dz()
        section.update_tensions()
        
        # section.update_span()
        section.update_span()
        force_vector = section.vector_force()

        norm_d_param = np.abs(np.linalg.norm(force_vector)**2 - mem**2)
                
        print("**"*10)
        print(compteur)
        # print(correction)
        print("force vector norm: ", np.linalg.norm(force_vector)**2)
        print(f"{norm_d_param=}")
        # print("-"*10)
        # print(section.nodes.dx)
        # print(section.nodes.dz)

        
        if norm_d_param < 100_000:
            section.finition = True
        if norm_d_param < .1:
            print("--end--"*10)
            print(norm_d_param)
            break
        if n_iter == compteur:
            print("max iteration reached")
            print(norm_d_param)
        
    print(f"force vector norm: {np.linalg.norm(force_vector)}")
    # ww = np.tile(aa, (5,1))
#     np.multiply(ww , np.array([[0,1,0,0,1]]).T)
    #   >> array([[0, 0, 0, 0, 0, 0, 0],
    #            [3, 2, 2, 2, 2, 2, 3],
    #            [0, 0, 0, 0, 0, 0, 0],
    #            [0, 0, 0, 0, 0, 0, 0],
    #            [3, 2, 2, 2, 2, 2, 3]])
    
    

def compute_derivative(section:Span, eps,):
    # force_vector = section._delta_dx(x0)
    mask_on_x = np.int32(section.nodes.ntype!=3)
    mask_on_z = np.int32(section.nodes.ntype!=2)
               
    dx_d = section._delta_dx(eps*mask_on_x)
    # section._delta_dx(eps)
    
    # force_vector = section._delta_dz(x0)           
    dz_d = section._delta_dz(eps*mask_on_z)
    # section._delta_dz(eps)
   
    return dx_d, dz_d
    
    
    





    
    
    
    