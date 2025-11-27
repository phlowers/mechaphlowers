# Benchmark mode

A functional benchmark mode is provided.

- Write your own benchmark scenario into test/test_benchmark.py
- mark the test with `@pytest.marks.benchmark`
- 2 ways to run:
  - manual: `make benchmark`
  - IDE integrated: modify in the pyproject.toml to get the pytest plugin display the benchmark.
  ```shell
  [tool.pytest.ini_options]
  addopts = ["-s", "-vv", "--durations=2", "-m", "benchmark"]
  ```
The result will be available in json and html format in the test_output folder.

