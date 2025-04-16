# copy from scipy fork https://github.com/mikofski/scipy/blob/master/scipy/optimize/tests/test_zeros.py
# why this code ? Scipy has been updated and cythonized. It is difficult to extract pieces of codes from recent code.
# Benchmarks shows that this code is 10x faster than a classical vectorized version of newton.
# And mechaphlowers needs a local newton method to avoid loading scipy.

# mypy / ruff is ignored here
#  type: ignore
#  ruff: noqa

from __future__ import absolute_import, division, print_function

from math import cos, exp, sin, sqrt

import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose, assert_equal, assert_warns

import mechaphlowers.core.numeric.numeric as zeros

# Import testing parameters
# from scipy.optimize._tstutils import functions, fstrings

TOL = 4 * np.finfo(float).eps  # tolerance


class TestBasic(object):
	def run_check(self, method, name):
		a = 0.5
		b = sqrt(3)
		xtol = rtol = TOL

	def test_newton(self):
		f1 = lambda x: x**2 - 2 * x - 1
		f1_1 = lambda x: 2 * x - 2
		f1_2 = lambda x: 2.0 + 0 * x

		f2 = lambda x: exp(x) - cos(x)
		f2_1 = lambda x: exp(x) + sin(x)
		f2_2 = lambda x: exp(x) + cos(x)

		for f, f_1, f_2 in [(f1, f1_1, f1_2), (f2, f2_1, f2_2)]:
			x = zeros.newton(f, 3, tol=1e-6)
			assert_allclose(f(x), 0, atol=1e-6)
			x = zeros.newton(f, 3, fprime=f_1, tol=1e-6)
			assert_allclose(f(x), 0, atol=1e-6)
			x = zeros.newton(f, 3, fprime=f_1, fprime2=f_2, tol=1e-6)
			assert_allclose(f(x), 0, atol=1e-6)

	def test_array_newton(self):
		"""test newton with array"""

		def f1(x, *a):
			b = a[0] + x * a[3]
			return a[1] - a[2] * (np.exp(b / a[5]) - 1.0) - b / a[4] - x

		def f1_1(x, *a):
			b = a[3] / a[5]
			return -a[2] * np.exp(a[0] / a[5] + x * b) * b - a[3] / a[4] - 1

		def f1_2(x, *a):
			b = a[3] / a[5]
			return -a[2] * np.exp(a[0] / a[5] + x * b) * b**2

		a0 = np.array(
			[
				5.32725221,
				5.48673747,
				5.49539973,
				5.36387202,
				4.80237316,
				1.43764452,
				5.23063958,
				5.46094772,
				5.50512718,
				5.42046290,
			]
		)
		a1 = (np.sin(range(10)) + 1.0) * 7.0
		args = (a0, a1, 1e-09, 0.004, 10, 0.27456)
		x0 = [7.0] * 10
		x = zeros.newton(f1, x0, f1_1, args)
		x_expected = (
			6.17264965,
			11.7702805,
			12.2219954,
			7.11017681,
			1.18151293,
			0.143707955,
			4.31928228,
			10.5419107,
			12.7552490,
			8.91225749,
		)
		assert_allclose(x, x_expected)
		# test halley's
		x = zeros.newton(f1, x0, f1_1, args, fprime2=f1_2)
		assert_allclose(x, x_expected)
		# test secant
		x = zeros.newton(f1, x0, args=args)
		assert_allclose(x, x_expected)

	def test_array_secant_active_zero_der(self):
		"""test secant doesn't continue to iterate zero derivatives"""
		x = zeros.newton(
			lambda x, *a: x * x - a[0],
			x0=[4.123, 5],
			args=[np.array([17, 25])],
		)
		assert_allclose(x, (4.123105625617661, 5.0))

	def test_array_newton_integers(self):
		# test secant with float
		x = zeros.newton(
			lambda y, z: z - y**2, [4.0] * 2, args=([15.0, 17.0],)
		)
		assert_allclose(x, (3.872983346207417, 4.123105625617661))
		# test integer becomes float
		x = zeros.newton(lambda y, z: z - y**2, [4] * 2, args=([15, 17],))
		assert_allclose(x, (3.872983346207417, 4.123105625617661))

	def test_array_newton_zero_der_failures(self):
		# test derivative zero warning
		assert_warns(
			RuntimeWarning,
			zeros.newton,
			lambda y: y**2 - 2,
			[0.0, 0.0],
			lambda y: 2 * y,
		)
		# test failures and zero_der
		with pytest.warns(RuntimeWarning):
			results = zeros.newton(
				lambda y: y**2 - 2,
				[0.0, 0.0],
				lambda y: 2 * y,
				full_output=True,
			)
			assert_allclose(results.root, 0)
			assert results.zero_der.all()
			assert not results.converged.any()

	def test_newton_full_output(self):
		# Test the full_output capability, both when converging and not.
		# Use simple polynomials, to avoid hitting platform dependencies
		# (e.g. exp & trig) in number of iterations
		f1 = lambda x: x**2 - 2 * x - 1  # == (x-1)**2 - 2
		f1_1 = lambda x: 2 * x - 2
		f1_2 = lambda x: 2.0 + 0 * x

		x0 = 3
		expected_counts = [(6, 7), (5, 10), (3, 9)]

		for derivs in range(3):
			kwargs = {
				'tol': 1e-6,
				'full_output': True,
			}
			for k, v in [['fprime', f1_1], ['fprime2', f1_2]][:derivs]:
				kwargs[k] = v

			x, r = zeros.newton(f1, x0, disp=False, **kwargs)
			assert_(r.converged)
			assert_equal(x, r.root)
			assert_equal(
				(r.iterations, r.function_calls), expected_counts[derivs]
			)
			if derivs == 0:
				assert r.function_calls <= r.iterations + 1
			else:
				assert_equal(r.function_calls, (derivs + 1) * r.iterations)

			# Now repeat, allowing one fewer iteration to force convergence failure
			iters = r.iterations - 1
			x, r = zeros.newton(f1, x0, maxiter=iters, disp=False, **kwargs)
			assert_(not r.converged)
			assert_equal(x, r.root)
			assert_equal(r.iterations, iters)

			if derivs == 1:
				# Check that the correct Exception is raised and
				# validate the start of the message.
				with pytest.raises(
					RuntimeError,
					match='Failed to converge after %d iterations, value is .*'
					% (iters),
				):
					x, r = zeros.newton(
						f1, x0, maxiter=iters, disp=True, **kwargs
					)

	def test_deriv_zero_warning(self):
		func = lambda x: x**2 - 2.0
		dfunc = lambda x: 2 * x
		assert_warns(
			RuntimeWarning, zeros.newton, func, 0.0, dfunc, disp=False
		)
		with pytest.raises(RuntimeError, match='Derivative was zero'):
			result = zeros.newton(func, 0.0, dfunc)

	def test_newton_does_not_modify_x0(self):
		# https://github.com/scipy/scipy/issues/9964
		x0 = np.array([0.1, 3])
		x0_copy = x0.copy()  # Copy to test for equality.
		zeros.newton(np.sin, x0, np.cos)
		np.testing.assert_array_equal(x0, x0_copy)


class TestRootResults:
	def test_repr(self):
		r = zeros.RootResults(
			root=1.0, iterations=44, function_calls=46, flag=0
		)
		expected_repr = (
			"      converged: True\n           flag: 'converged'"
			"\n function_calls: 46\n     iterations: 44\n"
			"           root: 1.0"
		)
		assert_equal(repr(r), expected_repr)


def test_complex_halley():
	"""Test Halley's works with complex roots"""

	def f(x, *a):
		return a[0] * x**2 + a[1] * x + a[2]

	def f_1(x, *a):
		return 2 * a[0] * x + a[1]

	def f_2(x, *a):
		retval = 2 * a[0]
		try:
			size = len(x)
		except TypeError:
			return retval
		else:
			return [retval] * size

	z = complex(1.0, 2.0)
	coeffs = (2.0, 3.0, 4.0)
	y = zeros.newton(f, z, args=coeffs, fprime=f_1, fprime2=f_2, tol=1e-6)
	# (-0.75000000000000078+1.1989578808281789j)
	assert_allclose(f(y, *coeffs), 0, atol=1e-6)
	z = [z] * 10
	coeffs = (2.0, 3.0, 4.0)
	y = zeros.newton(f, z, args=coeffs, fprime=f_1, fprime2=f_2, tol=1e-6)
	assert_allclose(f(y, *coeffs), 0, atol=1e-6)


def test_zero_der_nz_dp():
	"""Test secant method with a non-zero dp, but an infinite newton step"""
	# pick a symmetrical functions and choose a point on the side that with dx
	# makes a secant that is a flat line with zero slope, EG: f = (x - 100)**2,
	# which has a root at x = 100 and is symmetrical around the line x = 100
	# we have to pick a really big number so that it is consistently true
	# now find a point on each side so that the secant has a zero slope
	dx = np.finfo(float).eps ** 0.33
	# 100 - p0 = p1 - 100 = p0 * (1 + dx) + dx - 100
	# -> 200 = p0 * (2 + dx) + dx
	p0 = (200.0 - dx) / (2.0 + dx)
	x = zeros.newton(lambda y: (y - 100.0) ** 2, x0=[p0] * 10)
	assert_allclose(x, [100] * 10)


def test_array_newton_failures():
	"""Test that array newton fails as expected"""
	# p = 0.68  # [MPa]
	# dp = -0.068 * 1e6  # [Pa]
	# T = 323  # [K]
	diameter = 0.10  # [m]
	# L = 100  # [m]
	roughness = 0.00015  # [m]
	rho = 988.1  # [kg/m**3]
	mu = 5.4790e-04  # [Pa*s]
	u = 2.488  # [m/s]
	reynolds_number = rho * u * diameter / mu  # Reynolds number

	def colebrook_eqn(darcy_friction, re, dia):
		return 1 / np.sqrt(darcy_friction) + 2 * np.log10(
			roughness / 3.7 / dia + 2.51 / re / np.sqrt(darcy_friction)
		)

	# only some failures
	with pytest.warns(RuntimeWarning):
		result = zeros.newton(
			colebrook_eqn,
			x0=[0.01, 0.2, 0.02223, 0.3],
			maxiter=2,
			args=[reynolds_number, diameter],
			full_output=True,
		)
		assert not result.converged.all()
	# they all fail
	with pytest.raises(RuntimeError):
		result = zeros.newton(
			colebrook_eqn,
			x0=[0.01] * 2,
			maxiter=2,
			args=[reynolds_number, diameter],
			full_output=True,
		)


# this test should **not** raise a RuntimeWarning
def test_gh8904_zeroder_at_root_fails():
	"""Test that Newton or Halley don't warn if zero derivative at root"""

	# a function that has a zero derivative at it's root
	def f_zeroder_root(x):
		return x**3 - x**2

	# should work with secant
	r = zeros.newton(f_zeroder_root, x0=0)
	assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
	# test again with array
	r = zeros.newton(f_zeroder_root, x0=[0] * 10)
	assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)

	# 1st derivative
	def fder(x):
		return 3 * x**2 - 2 * x

	# 2nd derivative
	def fder2(x):
		return 6 * x - 2

	# should work with newton and halley
	r = zeros.newton(f_zeroder_root, x0=0, fprime=fder)
	assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
	r = zeros.newton(f_zeroder_root, x0=0, fprime=fder, fprime2=fder2)
	assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
	# test again with array
	r = zeros.newton(f_zeroder_root, x0=[0] * 10, fprime=fder)
	assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
	r = zeros.newton(f_zeroder_root, x0=[0] * 10, fprime=fder, fprime2=fder2)
	assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)

	# also test that if a root is found we do not raise RuntimeWarning even if
	# the derivative is zero, EG: at x = 0.5, then fval = -0.125 and
	# fder = -0.25 so the next guess is 0.5 - (-0.125/-0.5) = 0 which is the
	# root, but if the solver continued with that guess, then it will calculate
	# a zero derivative, so it should return the root w/o RuntimeWarning
	r = zeros.newton(f_zeroder_root, x0=0.5, fprime=fder)
	assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
	# test again with array
	r = zeros.newton(f_zeroder_root, x0=[0.5] * 10, fprime=fder)
	assert_allclose(r, 0, atol=zeros._xtol, rtol=zeros._rtol)
	# doesn't apply to halley


def test_gh_9608_preserve_array_shape():
	"""
	Test that shape is preserved for array inputs even if fprime or fprime2 is
	scalar
	"""

	def f(x):
		return x**2

	def fp(x):
		return 2 * x

	def fpp(x):
		return 2

	x0 = np.array([-2], dtype=np.float32)
	rt, r = zeros.newton(f, x0, fprime=fp, fprime2=fpp, full_output=True)
	assert r.converged

	x0_array = np.array([-2, -3], dtype=np.float32)
	# This next invocation should fail
	with pytest.raises(IndexError):
		result = zeros.newton(
			f, x0_array, fprime=fp, fprime2=fpp, full_output=True
		)

	def fpp_array(x):
		return 2 * np.ones(np.shape(x), dtype=np.float32)

	result = zeros.newton(
		f, x0_array, fprime=fp, fprime2=fpp_array, full_output=True
	)
	assert result.converged.all()


def test_gh9551_raise_error_if_disp_true():
	"""Test that if disp is true then zero derivative raises RuntimeError"""

	def f(x):
		return x * x + 1

	def f_p(x):
		return 2 * x

	assert_warns(RuntimeWarning, zeros.newton, f, 1.0, f_p, disp=False)
	with pytest.raises(
		RuntimeError,
		match=r'^Derivative was zero\. Failed to converge after \d+ iterations, value is [+-]?\d*\.\d+\.$',
	):
		result = zeros.newton(f, 1.0, f_p)
	root = zeros.newton(f, complex(10.0, 10.0), f_p)
	assert_allclose(root, complex(0.0, 1.0))


def test_gh_8881():
	r"""Test that Halley's method realizes that the 2nd order adjustment
	is too big and drops off to the 1st order adjustment."""
	n = 9

	def f(x):
		return np.power(x, 1.0 / n) - np.power(n, 1.0 / n)

	def fp(x):
		return np.power(x, (1.0 - n) / n) / n

	def fpp(x):
		return np.power(x, (1.0 - 2 * n) / n) * (1.0 / n) * (1.0 - n) / n

	x0 = 0.1
	# The root is at x=9.
	# The function has positive slope, x0 < root.
	# Newton succeeds in 8 iterations
	rt, r = zeros.newton(f, x0, fprime=fp, full_output=True)
	assert r.converged
	# Before the Issue 8881/PR 8882, halley would send x in the wrong direction.
	# Check that it now succeeds.
	rt, r = zeros.newton(f, x0, fprime=fp, fprime2=fpp, full_output=True)
	assert r.converged
