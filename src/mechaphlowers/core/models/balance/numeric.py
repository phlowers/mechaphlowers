# from https://github.com/NKrvavica/fqs/tree/master MIT Licence

import math

import numpy as np


def multi_cubic(a0, b0, c0, d0, all_roots=True):
    '''Analytical closed-form solver for multiple cubic equations
    (3rd order polynomial), based on `numpy` functions.

    Parameters
    ----------
    a0, b0, c0, d0: array_like
        Input data are coefficients of the Cubic polynomial::

            a0*x^3 + b0*x^2 + c0*x + d0 = 0

    all_roots: bool, optional
        If set to `True` (default) all three roots are computed and returned.
        If set to `False` only one (real) root is computed and returned.

    Returns
    -------
    roots: ndarray
        Output data is an array of three roots of given polynomials of size
        (3, M) if `all_roots=True`, and an array of one root of size (M,)
        if `all_roots=False`.
    '''

    ''' Reduce the cubic equation to to form:
        x^3 + a*x^2 + bx + c = 0'''
    a, b, c = b0 / a0, c0 / a0, d0 / a0

    # Some repeating constants and variables
    third = 1.0 / 3.0
    a13 = a * third
    a2 = a13 * a13
    sqr3 = math.sqrt(3)

    # Additional intermediate variables
    f = third * b - a2
    g = a13 * (2 * a2 - b) + c
    h = 0.25 * g * g + f * f * f

    # Masks for different combinations of roots
    m1 = (f == 0) & (g == 0) & (h == 0)  # roots are real and equal
    m2 = (~m1) & (h <= 0)  # roots are real and distinct
    m3 = (~m1) & (~m2)  # one real root and two complex

    def cubic_root(x):
        '''Compute cubic root of a number while maintaining its sign'''
        root = np.zeros_like(x)
        positive = x >= 0
        negative = ~positive
        root[positive] = x[positive] ** third
        root[negative] = -((-x[negative]) ** third)
        return root

    def roots_all_real_equal(c):
        '''Compute cubic roots if all roots are real and equal'''
        r1 = -cubic_root(c)
        if all_roots:
            return r1, r1, r1
        else:
            return r1

    def roots_all_real_distinct(a13, f, g, h):
        '''Compute cubic roots if all roots are real and distinct'''
        j = np.sqrt(-f)
        k = np.arccos(-0.5 * g / (j * j * j))
        m = np.cos(third * k)
        r1 = 2 * j * m - a13
        if all_roots:
            n = sqr3 * np.sin(third * k)
            r2 = -j * (m + n) - a13
            r3 = -j * (m - n) - a13
            return r1, r2, r3
        else:
            return r1

    def roots_one_real(a13, g, h):
        '''Compute cubic roots if one root is real and other two are complex'''
        sqrt_h = np.sqrt(h)
        S = cubic_root(-0.5 * g + sqrt_h)
        U = cubic_root(-0.5 * g - sqrt_h)
        S_plus_U = S + U
        r1 = S_plus_U - a13
        if all_roots:
            S_minus_U = S - U
            r2 = -0.5 * S_plus_U - a13 + S_minus_U * sqr3 * 0.5j
            r3 = -0.5 * S_plus_U - a13 - S_minus_U * sqr3 * 0.5j
            return r1, r2, r3
        else:
            return r1

    # Compute roots
    if all_roots:
        roots = np.zeros((3, len(a))).astype(complex)
        roots[:, m1] = roots_all_real_equal(c[m1])
        roots[:, m2] = roots_all_real_distinct(a13[m2], f[m2], g[m2], h[m2])
        roots[:, m3] = roots_one_real(a13[m3], g[m3], h[m3])
    else:
        roots = np.zeros(len(a))  # .astype(complex)
        roots[m1] = roots_all_real_equal(c[m1])
        roots[m2] = roots_all_real_distinct(a13[m2], f[m2], g[m2], h[m2])
        roots[m3] = roots_one_real(a13[m3], g[m3], h[m3])

    return roots


def cubic_roots(p, only_max_real=True):
    '''
    A caller function for a fast cubic root solver (3rd order polynomial).

    If a single cubic equation or a set of fewer than 100 equations is
    given as an input, this function will call `single_cubic` inside
    a list comprehension. Otherwise (if a more than 100 equtions is given), it
    will call `multi_cubic` which is based on `numpy` functions.
    Both equations are based on a closed-form analytical solutions by Cardano.

    Parameters
    ----------
    p: array_like
        Input data are coefficients of the Cubic polynomial of the form:

            p[0]*x^3 + p[1]*x^2 + p[2]*x + p[3] = 0

        Stacked arrays of coefficient are allowed, which means that ``p`` may
        have size ``(4,)`` or ``(M, 4)``, where ``M>0`` is the
        number of polynomials. Note that the first axis should be used for
        stacking.

    Returns
    -------
    roots: ndarray
        Output data is an array of three roots of given polynomials,
        of size ``(M, 3)``.

    Examples
    --------
    >>> roots = cubic_roots([1, 7, -806, -1050])
    >>> roots
    array([[ 25.80760451+0.j, -31.51667909+0.j,  -1.29092543+0.j]])

    >>> roots = cubic_roots([1, 2, 3, 4])
    >>> roots
    array([[-1.65062919+0.j        , -0.1746854 +1.54686889j,
            -0.1746854 -1.54686889j]])

    >>> roots = cubic_roots([[1, 2, 3, 4],
                               [1, 7, -806, -1050]])
    >>> roots
    array([[ -1.65062919+0.j        ,  -0.1746854 +1.54686889j,
             -0.1746854 -1.54686889j],
           [ 25.80760451+0.j        , -31.51667909+0.j        ,
             -1.29092543+0.j        ]])
    '''
    # Convert input to array (if input is a list or tuple)
    p = np.asarray(p)

    # If only one set of coefficients is given, add axis
    if p.ndim < 2:
        p = p[np.newaxis, :]

    # Check if four coefficients are given
    if p.shape[1] != 4:
        raise ValueError(
            'Expected 3rd order polynomial with 4 '
            'coefficients, got {:d}.'.format(p.shape[1])
        )

    # not for us :
    # if p.shape[0] < 100:
    #     roots = [single_cubic(*pi) for pi in p]
    #     return np.array(roots)
    # else:
    # Add for us : transpose to get 3 columns (solutions) and n rows
    roots = multi_cubic(*p.T).T

    # add for us a preprocessing step to remove imaginary part a take only maximum root
    if only_max_real is True:
        roots = np.where(np.isreal(roots), roots, 0)
        roots = np.max(roots, axis=1)
    return roots
