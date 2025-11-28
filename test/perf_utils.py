import time

import numpy as np

from mechaphlowers.utils import numpy_cache


# Re-implement the formula used in CatenarySpan.compute_x_m to benchmark
def compute_x_m_impl(
    a: np.ndarray, b: np.ndarray, p: np.ndarray
) -> np.ndarray:
    """Vectorized implementation of compute_x_m used by CatenarySpan."""
    return -a / 2 + p * np.arcsinh(b / (2 * p * np.sinh(a / (2 * p))))


# Decorated variants
@numpy_cache
def comp_cached(a: np.ndarray, b: np.ndarray, p: np.ndarray):
    return compute_x_m_impl(a, b, p)


def bench_once(func, *args, repeat=3):
    """Run func once and return elapsed seconds (averaged over `repeat` runs)."""
    # do a few short runs to stabilize
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = func(*args)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)


def main():
    rng = np.random.default_rng(0)

    # choose a realistic size — tweak n to make compute significant on your machine
    n = 100

    # realistic ranges for the parameters
    a = rng.uniform(400.0, 1200.0, size=n)  # span lengths (meters)
    b = rng.uniform(-20.0, 20.0, size=n)  # elevation diff (meters)
    p = rng.uniform(500.0, 2500.0, size=n)  # sagging parameter

    b1 = rng.uniform(-20.0, 20.0, size=n)  # elevation diff (meters)
    p1 = rng.uniform(500.0, 2500.0, size=n)  # sagging parameter

    # increase the cache to see lookup table effects
    for i in range(1000):
        comp_cached(rng.uniform(400.0, 1200.0, size=n), b1, p1)

    print(f"Benchmarking with n={n} elements")

    # baseline: raw computation
    t_raw = bench_once(compute_x_m_impl, a, b, p, repeat=100)
    print(f"Raw compute_x_m_impl: {t_raw*1e6:.10f} us (avg)")

    # numpy_cache: first call (miss) then second call (hit)
    t_cached_miss = bench_once(comp_cached, a, b, p, repeat=2)
    t_cached_hit = bench_once(comp_cached, a, b, p, repeat=3)
    print(
        f"numpy_cache  - miss: {t_cached_miss*1e6:.10f} us, hit: {t_cached_hit*1e6:.10f} us, cache_size: {len(comp_cached._cache)}"
    )

    # Speedups
    hit_speedup = t_raw / t_cached_hit if t_cached_hit > 0 else float("inf")
    miss_overhead = t_cached_miss / t_raw if t_raw > 0 else float("inf")
    print(f"Speedup (raw / cached_hit): {hit_speedup:.1f}x")
    print(f"Miss overhead (cached_miss / raw): {miss_overhead:.2f}x")

    # Show that cache_clear actually removes entries
    comp_cached.cache_clear()
    print(f"After clear, cache_size: {len(comp_cached._cache)}")


if __name__ == "__main__":
    main()
