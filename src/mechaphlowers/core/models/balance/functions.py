


import numpy as np


def L(p, x_n, x_m):
    return p * (np.sinh(x_n / p) - np.sinh(x_m / p))

def x_m(a, b, p):
    return -a / 2 + p * np.arcsinh(b / (2 * p * np.sinh(a / (2 * p))))

def x_n(a, b, p):
    return x_m(a,b,p) + a

def z(x, p, x_m):
    # repeating value to perform multidim operation
    xx = x.T + x_m # >>> why + x_m?
    # self.p is a vector of size (nb support, ). I need to convert it in a matrix (nb support, 1) to perform matrix operation after.
    # Ex: self.p = array([20,20,20,20]) -> self.p([:,new_axis]) = array([[20],[20],[20],[20]])
    pp = p[:, np.newaxis]

    rr = pp * (np.cosh(xx / pp) - 1)

    # reshaping back to p,x -> (vertical, horizontal)
    return rr.T

def T_moy(p, L, x_n, x_m, lineic_weight):
    
    a = x_n - x_m


    return p * lineic_weight * (a + (np.sinh(2*x_n / p) - np.sinh(2*x_m / p))*p/2) / L /2


def z_from_x_2ddl(span):

    span.compute()
    span
    return b * np.sinh(x / p) / np.sinh(a / (2 * p))