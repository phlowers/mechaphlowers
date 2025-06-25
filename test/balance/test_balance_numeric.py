# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 11:14:52 2019

@author: NKrvavica
"""

# import timeit
import numpy as np
# import fqs
from mechaphlowers.core.models.balance.numeric import cubic_roots

def eig_cubic_roots(p):
    '''Finds cubic roots via numerical eigenvalue solver
    `npumpy.linalg.eigvals` from a 3x3 companion matrix'''
    a, b, c = (p[:, 1]/p[:, 0], p[:, 2]/p[:, 0], p[:, 3]/p[:, 0],)
    N = len(a)
    A = np.zeros((N, 3, 3))
    A[:, 1:, :2] = np.eye(2)
    A[:, :, 2] = - np.array([c, b, a]).T
    roots = np.linalg.eigvals(A)
    return roots


def test_cubic_roots():
    
    import plotly.graph_objects as go

    # fig = go.Figure()
    # x = np.arange(-5, 5, 0.01, dtype=float)
    # y = -1*x**3 -3*x**2 +2*x -0
    # fig.add_trace(go.Scatter(x=x,y=y))

    p0 = np.array([5, .1, .01, .00001,1e-10])
    p1 = np.array((2,2,0,0,0))
    p2 = np.array([-3, -3, -3, 0, 0])
    p3 = np.array([-4, -4,-4,-4,0])
    
    p = np.vstack( (p0, p1, p2, p3) ).T
    
    r = cubic_roots(p, only_max_real=False)
    
    np.testing.assert_array_almost_equal(r, np.array(
        [
            [  1.        +0.j, -0.7+0.55677644j,  -0.7-0.55677644j],
            [  2.17988479+0.j, -21.31917642+0.j,  -0.86070837+0.j],
            [ 17.95219749+0.j, -16.61081904+0.j,  -1.34137846+0.j],
            [ 73.68062997+0.j, -36.84031499+63.80929732j, -36.84031499-63.80929732j],
            [ -0.        +0.j,  -0.        +0.j,  -0.        +0.j]
            ]
        )
                                         )
    
    r = cubic_roots(p, only_max_real=True)
    np.testing.assert_array_almost_equal(r, np.array(
        [ 1. , 2.17988479, 17.95219749, 73.68062997, 0. ]
        )
    )

# fig.show()




# # --------------------------------------------------------------------------- #
# # Test speed of fqs cubic solver compared to np.roots and np.linalg.eigvals
# # --------------------------------------------------------------------------- #

# # Number of samples (sets of randomly generated cubic coefficients)
# N = 100

# # Generate polynomial coefficients
# range_coeff = 100
# p = np.random.rand(N, 4)*(range_coeff) - range_coeff/2

# # number of runs
# runs = 10

# times = []
# for i in range(runs):
#     start = timeit.default_timer()
#     roots1 = [np.roots(pi) for pi in p]
#     stop = timeit.default_timer()
#     time = stop - start
#     times.append(time)
# print('np.roots: {:.4f} ms (best of {} runs)'
#       .format(np.array(times).mean()*1_000, runs))

# times = []
# for i in range(runs):
#     start = timeit.default_timer()
#     roots2 = eig_cubic_roots(p)
#     stop = timeit.default_timer()
#     time = stop - start
#     times.append(time)
# print('np.linalg.eigvals: {:.4f} ms (average of {} runs)'
#       .format(np.array(times).mean()*1_000, runs))
# print('max err: {:.2e}'.format(abs(np.sort(roots2, axis=1)
#                     - (np.sort(roots1, axis=1))).max()))

# times = []
# for i in range(runs):
#     start = timeit.default_timer()
#     roots3 = [fqs.single_cubic(*pi) for pi in p]
#     stop = timeit.default_timer()
#     time = stop - start
#     times.append(time)
# print('fqs.single_cubic: {:.4f} ms (average of {} runs)'
#       .format(np.array(times).mean()*1_000, runs))
# print('max err: {:.2e}'.format(abs(np.sort(roots3, axis=1)
#                     - (np.sort(roots1, axis=1))).max()))

# times = []
# for i in range(runs):
#     start = timeit.default_timer()
#     roots = fqs.multi_cubic(*p.T)
#     roots4 = np.array(roots).T
#     stop = timeit.default_timer()
#     time = stop - start
#     times.append(time)
# print('fqs.multi_cubic: {:.4f} ms (average of {} runs)'
#       .format(np.array(times).mean()*1_000, runs))
# print('max err: {:.2e}'.format(abs(np.sort(roots4, axis=1)
#                     - (np.sort(roots1, axis=1))).max()))

# times = []
# for i in range(runs):
#     start = timeit.default_timer()
#     roots5 = fqs.cubic_roots(p)
#     stop = timeit.default_timer()
#     time = stop - start
#     times.append(time)
# print('fqs.cubic_roots: {:.4f} ms (average of {} runs)'
#       .format(np.array(times).mean()*1_000, runs))
# print('max err: {:.2e}'.format(abs(np.sort(roots5, axis=1)
#                     - (np.sort(roots1, axis=1))).max()))
# # --------------------------------------------------------------------------- #