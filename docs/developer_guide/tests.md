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

## Dependency compatibility

The supported dependency ranges include pandas 2 and 3, Plotly 5 and 6,
and xxhash 3 and 4. Plotly remains below 7 because the geographic line
map uses `Scattermapbox`, which was removed in Plotly 7. Supporting that
release requires a separate migration to MapLibre.

Validate the locked dependencies on both the minimum and newest tested
Python versions using uv-managed interpreters:

```bash
uv sync --managed-python --python 3.11 --all-groups --extra full
uv run --managed-python --python 3.11 --locked pytest
uv sync --managed-python --python 3.14 --all-groups --extra full
uv run --managed-python --python 3.14 --locked pytest
```

When widening dependency bounds, also test the affected code paths with
the retained older dependency majors. An isolated run can override the
locked versions without changing the lockfile:

```bash
uv run --managed-python --python 3.11 --locked --isolated \
    --with 'pandas==2.3.1' --with 'plotly==5.24.1' --with 'xxhash==3.5.0' \
    pytest test/entities/test_obstacle_array.py \
    test/core/geometry/test_position_engine.py test/test_utils.py test/plotting/
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