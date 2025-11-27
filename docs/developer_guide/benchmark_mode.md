# Benchmark mode

A functional benchmark mode is provided.

If manual data is needed, you can use the `input_data` folder

- Write your own benchmark scenario into test/benchmark
- mark the test with `@pytest.marks.benchmark`
- do not forget to add a skip or xfail if the test should fail / to improve tests execution speed.
- 2 ways to run:
  - manual: `make benchmark`
  - IDE integrated: comment the skip or xfail mark to get the pytest plugin display the benchmark and play.

The result will be available in json and html format in the output_test folder.

