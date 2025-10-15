# from https://github.com/NKrvavica/fqs/tree/master MIT Licence
"""
Created on Thu Jan  3 11:14:52 2019

@author: NKrvavica
"""

# import timeit
import numpy as np

# import fqs
from mechaphlowers.numeric.cubic import cubic_roots


def eig_cubic_roots(p):
    '''Finds cubic roots via numerical eigenvalue solver
    `npumpy.linalg.eigvals` from a 3x3 companion matrix'''
    a, b, c = (
        p[:, 1] / p[:, 0],
        p[:, 2] / p[:, 0],
        p[:, 3] / p[:, 0],
    )
    N = len(a)
    A = np.zeros((N, 3, 3))
    A[:, 1:, :2] = np.eye(2)
    A[:, :, 2] = -np.array([c, b, a]).T
    roots = np.linalg.eigvals(A)
    return roots


def test_cubic_roots():
    p0 = np.array([5, 0.1, 0.01, 0.00001, 1e-10])
    p1 = np.array((2, 2, 0, 0, 0))
    p2 = np.array([-3, -3, -3, 0, 0])
    p3 = np.array([-4, -4, -4, -4, 0])

    p = np.vstack((p0, p1, p2, p3)).T

    r = cubic_roots(p, only_max_real=False)

    np.testing.assert_array_almost_equal(
        r,
        np.array(
            [
                [1.0 + 0.0j, -0.7 + 0.55677644j, -0.7 - 0.55677644j],
                [2.17988479 + 0.0j, -21.31917642 + 0.0j, -0.86070837 + 0.0j],
                [17.95219749 + 0.0j, -16.61081904 + 0.0j, -1.34137846 + 0.0j],
                [
                    73.68062997 + 0.0j,
                    -36.84031499 + 63.80929732j,
                    -36.84031499 - 63.80929732j,
                ],
                [-0.0 + 0.0j, -0.0 + 0.0j, -0.0 + 0.0j],
            ]
        ),
    )

    r = cubic_roots(p, only_max_real=True)
    np.testing.assert_array_almost_equal(
        r, np.array([1.0, 2.17988479, 17.95219749, 73.68062997, 0.0])
    )
