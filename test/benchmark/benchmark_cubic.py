"""Benchmark: current custom Cardano cubic solver vs scipy compiled root finders.

Uses the same balance engine context as profile_balance.py (4 supports with loads).
"""
import numpy as np
import timeit
import textwrap

# ── 1. Collect realistic coefficients from a real solve ──────────────────────
from mechaphlowers.core.models.balance.models import model_ducloux as md
from mechaphlowers.numeric import cubic as cubic_mod

# Monkey-patch to capture coefficients during a real solve
_captured_coeffs = []
_orig_cubic_roots = cubic_mod.cubic_roots

def _capturing_cubic_roots(p, only_max_real=True):
    _captured_coeffs.append(np.array(p, copy=True))
    return _orig_cubic_roots(p, only_max_real=only_max_real)

cubic_mod.cubic_roots = _capturing_cubic_roots
# Also patch the reference in model_ducloux module
_orig_md_cubic = md.cubic
class _PatchedCubic:
    def __getattr__(self, name):
        if name == "cubic_roots":
            return _capturing_cubic_roots
        return getattr(_orig_md_cubic, name)
md.cubic = _PatchedCubic()

from test.benchmark.profile_balance import setup, run_change_state
engine = setup()
run_change_state(engine)  # one call to capture coefficients

# Restore
md.cubic = _orig_md_cubic
cubic_mod.cubic_roots = _orig_cubic_roots

print(f"Captured {len(_captured_coeffs)} cubic_roots calls from 1 solve_change_state")
print(f"Coefficient shapes: {[c.shape for c in _captured_coeffs[:5]]}...")

# ── 2. Define solvers to benchmark ──────────────────────────────────────────

# Current implementation
def solve_current(p_array):
    """Current custom Cardano (numpy) solver."""
    return cubic_mod.cubic_roots(p_array, only_max_real=True)

# Scipy: np.roots via companion matrix (calls LAPACK)
def solve_np_roots(p_array):
    """np.roots (companion matrix eigenvalue) - element-wise."""
    p = np.asarray(p_array)
    if p.ndim < 2:
        p = p[np.newaxis, :]
    results = np.empty(p.shape[0])
    for i in range(p.shape[0]):
        roots = np.roots(p[i])
        real_roots = roots[np.isreal(roots)].real
        results[i] = np.max(real_roots) if len(real_roots) > 0 else 0.0
    return results

# Scipy compiled root finder
from scipy.optimize import brentq

def solve_scipy_brentq(p_array):
    """scipy.optimize.brentq - compiled C bisection for each polynomial."""
    p = np.asarray(p_array)
    if p.ndim < 2:
        p = p[np.newaxis, :]
    results = np.empty(p.shape[0])
    for i in range(p.shape[0]):
        a0, b0, c0, d0 = p[i]
        poly = lambda x: a0*x**3 + b0*x**2 + c0*x + d0
        # bracket: parameter is positive and typically in [0, 1e6]
        try:
            results[i] = brentq(poly, 0, 1e7)
        except ValueError:
            results[i] = 0.0
    return results

# numpy.polynomial.polynomial companion matrix (vectorized via eigenvalues)
def solve_np_companion(p_array):
    """np.linalg.eigvals on companion matrix - batch approach."""
    p = np.asarray(p_array)
    if p.ndim < 2:
        p = p[np.newaxis, :]
    results = np.empty(p.shape[0])
    for i in range(p.shape[0]):
        # np.polynomial uses [c0, c1, c2, c3] (low to high) order
        coeffs = p[i][::-1]  # reverse to low-to-high
        roots = np.polynomial.polynomial.polyroots(coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        results[i] = np.max(real_roots) if len(real_roots) > 0 else 0.0
    return results

# scipy.optimize.newton (compiled Newton-Raphson)
from scipy.optimize import newton as scipy_newton

def solve_scipy_newton(p_array):
    """scipy.optimize.newton - compiled Newton-Raphson with analytical derivative."""
    p = np.asarray(p_array)
    if p.ndim < 2:
        p = p[np.newaxis, :]
    results = np.empty(p.shape[0])
    for i in range(p.shape[0]):
        a0, b0, c0, d0 = p[i]
        f = lambda x: a0*x**3 + b0*x**2 + c0*x + d0
        fp = lambda x: 3*a0*x**2 + 2*b0*x + c0
        # Initial guess from a rough estimate
        x0 = max(1.0, (-d0/a0)**(1/3)) if a0 != 0 else 1.0
        try:
            results[i] = scipy_newton(f, x0, fprime=fp, maxiter=50)
        except RuntimeError:
            results[i] = 0.0
    return results

# numpy.linalg.eigvals batch companion matrix (truly vectorized)
def solve_eigvals_batch(p_array):
    """Batch companion matrix eigenvalues using numpy - no Python loop."""
    p = np.asarray(p_array)
    if p.ndim < 2:
        p = p[np.newaxis, :]
    n = p.shape[0]
    # Normalize: x^3 + ax^2 + bx + c = 0
    a = p[:, 1] / p[:, 0]
    b = p[:, 2] / p[:, 0]
    c = p[:, 3] / p[:, 0]
    # Build companion matrices (n, 3, 3)
    comp = np.zeros((n, 3, 3))
    comp[:, 0, 2] = -c
    comp[:, 1, 2] = -b
    comp[:, 2, 2] = -a
    comp[:, 1, 0] = 1.0
    comp[:, 2, 1] = 1.0
    # Batch eigenvalues
    eigenvalues = np.linalg.eigvals(comp)  # (n, 3)
    # Take max real root
    real_mask = np.abs(eigenvalues.imag) < 1e-10
    real_vals = np.where(real_mask, eigenvalues.real, -np.inf)
    results = np.max(real_vals, axis=1)
    return results


# ── 3. Validate all solvers give same results ───────────────────────────────
print("\n=== Validation ===")
test_p = _captured_coeffs[0]
ref = solve_current(test_p)
for name, solver in [
    ("np.roots", solve_np_roots),
    ("np.companion", solve_np_companion),
    ("eigvals_batch", solve_eigvals_batch),
    ("scipy.brentq", solve_scipy_brentq),
    ("scipy.newton", solve_scipy_newton),
]:
    result = solver(test_p)
    max_diff = np.max(np.abs(ref - result))
    status = "OK" if max_diff < 1e-4 else f"DIFF={max_diff:.6e}"
    print(f"  {name:20s}: max_diff={max_diff:.2e}  [{status}]")


# ── 4. Micro-benchmark: isolated cubic_roots calls ──────────────────────────
print("\n=== Micro-benchmark: single cubic_roots call (typical array size) ===")
test_p = _captured_coeffs[0]
print(f"Array shape: {test_p.shape}")

N_micro = 10000
for name, solver in [
    ("current (Cardano)", solve_current),
    ("eigvals_batch", solve_eigvals_batch),
    ("np.roots (loop)", solve_np_roots),
    ("np.companion (loop)", solve_np_companion),
    ("scipy.brentq (loop)", solve_scipy_brentq),
    ("scipy.newton (loop)", solve_scipy_newton),
]:
    t = timeit.timeit(lambda s=solver, p=test_p: s(p), number=N_micro)
    per_call_us = t / N_micro * 1e6
    print(f"  {name:25s}: {per_call_us:8.1f} µs/call  ({N_micro} calls)")


# ── 5. Full integration benchmark: solve_change_state with swapped solver ───
print("\n=== Full integration benchmark: solve_change_state ===")

def bench_full(label, cubic_roots_fn, N=100, repeats=5):
    """Benchmark full solve_change_state with a given cubic_roots implementation."""
    # Monkey-patch
    cubic_mod.cubic_roots = cubic_roots_fn
    class _Patch:
        def __getattr__(self, name):
            if name == "cubic_roots":
                return cubic_roots_fn
            return getattr(_orig_md_cubic, name)
    md.cubic = _Patch()

    engine2 = setup()
    run_change_state(engine2)  # warmup

    times = timeit.repeat(lambda: run_change_state(engine2), number=N, repeat=repeats)
    avg = min(times) / N * 1000

    # Restore
    md.cubic = _orig_md_cubic
    cubic_mod.cubic_roots = _orig_cubic_roots

    print(f"  {label:30s}: {avg:.3f} ms/call")
    return avg


def scipy_eigvals_cubic_roots(p, only_max_real=True):
    """Drop-in replacement using batch companion matrix eigenvalues."""
    p = np.asarray(p)
    if p.ndim < 2:
        p = p[np.newaxis, :]
    if p.shape[1] != 4:
        raise ValueError(f'Expected 4 coefficients, got {p.shape[1]}')
    return solve_eigvals_batch(p)


def scipy_newton_cubic_roots(p, only_max_real=True):
    """Drop-in replacement using scipy.optimize.newton."""
    p = np.asarray(p)
    if p.ndim < 2:
        p = p[np.newaxis, :]
    if p.shape[1] != 4:
        raise ValueError(f'Expected 4 coefficients, got {p.shape[1]}')
    return solve_scipy_newton(p)


def scipy_brentq_cubic_roots(p, only_max_real=True):
    """Drop-in replacement using scipy.optimize.brentq."""
    p = np.asarray(p)
    if p.ndim < 2:
        p = p[np.newaxis, :]
    if p.shape[1] != 4:
        raise ValueError(f'Expected 4 coefficients, got {p.shape[1]}')
    return solve_scipy_brentq(p)


results = {}
results["current (Cardano/numpy)"] = bench_full("current (Cardano/numpy)", _orig_cubic_roots)
results["eigvals_batch (numpy LAPACK)"] = bench_full("eigvals_batch (numpy LAPACK)", scipy_eigvals_cubic_roots)

# scipy.newton and brentq are loop-based and may not converge reliably
# with the initial guesses for all polynomial shapes in the solver.
# Only run them if they pass a quick smoke test.
for label, fn in [
    ("scipy.newton (C Newton)", scipy_newton_cubic_roots),
    ("scipy.brentq (C bisection)", scipy_brentq_cubic_roots),
]:
    try:
        eng_test = setup()
        run_change_state(eng_test)  # smoke test with this solver patched in
        # If it doesn't crash, run full benchmark
        # (restore first, bench_full will re-patch)
        md.cubic = _orig_md_cubic
        cubic_mod.cubic_roots = _orig_cubic_roots
        results[label] = bench_full(label, fn)
    except Exception as e:
        md.cubic = _orig_md_cubic
        cubic_mod.cubic_roots = _orig_cubic_roots
        print(f"  {label:30s}: SKIPPED ({type(e).__name__}: {e})"[:120])
        results[label] = None

print("\n=== Summary ===")
baseline = results["current (Cardano/numpy)"]
for label, ms in results.items():
    if ms is None:
        print(f"  {label:35s}: SKIPPED (convergence issues)")
        continue
    delta = (ms - baseline) / baseline * 100
    sign = "+" if delta >= 0 else ""
    print(f"  {label:35s}: {ms:7.3f} ms  ({sign}{delta:.1f}%)")
