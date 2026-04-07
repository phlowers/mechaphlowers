# Tests

The test suite is based on `pytest` and is located in the `test/` folder.

## Running tests

- Run all tests:

```bash
pytest
```

- Run one file:

```bash
pytest test/path/to/test_file.py
```

- Run one test:

```bash
pytest test/path/to/test_file.py::test_name
```

## Fixtures and shared test setup

Shared fixtures are defined in `test/conftest.py` (data builders, common objects, report hooks, and helper toggles).

## `SHOW_FIGURES` mechanism

Some tests can display Plotly figures for manual inspection. This behavior is controlled by the `show_figures` fixture in `test/conftest.py`.

The fixture reads the `SHOW_FIGURES` environment variable and enables figure display when it is set to one of:

- `1`
- `true`
- `yes`

Examples:

```bash
SHOW_FIGURES=1 pytest
```

```bash
SHOW_FIGURES=true pytest test/path/to/test_file.py
```

When `SHOW_FIGURES` is not set (or set to another value), figures are not shown.

## VS Code test debugging configuration

When debugging tests from VS Code, environment variables from your shell are not always propagated to the test debug session.

To ensure `SHOW_FIGURES` is available, add a `debug-test` launch configuration in `.vscode/launch.json`:

```json
{
	"name": "Debug Unit Test",
	"type": "python",
	"request": "launch",
	"justMyCode": false,
	"purpose": ["debug-test"],
	"env": {
		"PYDEVD_WARN_EVALUATION_TIMEOUT": "500",
		"SHOW_FIGURES": "True"
	}
}
```

With this configuration, launching tests in debug mode from VS Code will honor the `SHOW_FIGURES` mechanism.