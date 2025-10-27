# Mechaphlowers

[![PyPI Latest Release](https://img.shields.io/pypi/v/mechaphlowers.svg)](https://pypi.org/project/mechaphlowers/)
[![MPL-2.0 License](https://img.shields.io/badge/license-MPL_2.0-blue.svg)](https://www.mozilla.org/en-US/MPL/2.0/)
[![versions](https://img.shields.io/badge/python-3.11%7C3.12-blue)](https://github.com/phlowers/mechaphlowers)
[![Documentation](https://readthedocs.org/projects/mechaphlowers/badge/?version=latest)](https://phlowers.readthedocs.io/projects/mechaphlowers/en/latest)
[![Actions Status](https://github.com/phlowers/mechaphlowers/actions/workflows/dev-ci.yml/badge.svg)](https://github.com/phlowers/mechaphlowers/actions)

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=phlowers_mechaphlowers&metric=alert_status)](https://sonarcloud.io/dashboard?id=phlowers_mechaphlowers) [![Coverage](https://sonarcloud.io/api/project_badges/measure?project=phlowers_mechaphlowers&metric=coverage)](https://sonarcloud.io/dashboard?id=phlowers_mechaphlowers)

[![Numpy](https://img.shields.io/badge/numpy-v2-blue)](https://numpy.org/)
[![pyodide](https://img.shields.io/badge/works_on-pyodide-%237303fc)](https://pyodide.org/en/stable/index.html)
[![lite-badge](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://phlowers.github.io/phlowers-notebooks/lab/index.html)

Physical calculation package for the mechanics and geometry of overhead power lines.

## User

### Environment

Mechaphlowers is using uv for project, python version and dependencies management. You can use uv which is very similar to pip. You can also use other tools compatible with pip.

See [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) to install it.


You need a compatible python version. You may have to install it manually (e.g. with pyenv).
Then you may create a virtualenv, install dependencies and activate the env:

```console
    uv venv --python 3.11
    source .venv/bin/activate
```

!!! Tip

    You would probably use an editor, make sure you configure it to use the same virtual environment you created (it will probably autodetect it) so that you can get autocompletion and inline errors. Here some links for [VSCode](https://code.visualstudio.com/docs/python/environments#_select-and-activate-an-environment) and [PyCharm](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html).  

### Set up mechaphlowers

Install the package.
```console
    uv pip install mechaphlowers
```

Use it ! You can report to the user guide section or go to our tutorials [notebook jupyter server](https://phlowers.github.io/phlowers-notebooks/lab/index.html) to try it.

```python
    import mechaphlowers as mph
    print(mph.__version__)
```

## Developers

### Environment

You need to install the project with all the development and documentation packages:

```console
    uv venv --python 3.11
    source .venv/bin/activate
    uv sync --group all
```

### Checks and rules

#### Format and linter

Once dev dependencies are installed, you may format and lint python files like this:

```console
    uv run ruff format
    uv run ruff check
```

Use following command if you only want to check if files are correctly formatted:

```console
    uv run ruff format --check
```

You may automatically fix some linting errors:

```console
    uv run ruff check --fix
```

Tip: if using VSCode/VSCodium, you may also use Ruff extension.

#### How to check typing

In order to check type hints consistency, you may run:

```console
    uv run mypy .
```

#### How to test

```console
    uv run coverage run -m pytest
    uv run coverage report
```

#### Run all checks in one

A Makefile provide a fast access to those different checks.  
You may run every check mentioned above with just one command:

```console
    make all
```

### Requirements

#### Lock file

The generation of the lock file is important.  
Do not forget to update it with:

```console
    uv lock
    uv lock --check # to check if changes have been done
```

#### Installation from lock file only

When syncing, uv can update the lock file. But it can be an unwanted behavior. In this case use:
```console
    uv sync --frozen --group all
```

#### Pip compatibility

Requirements can be extracted with `pip compile`. See [here](https://docs.astral.sh/uv/pip/compile/#locking-requirements) for more information.


### Build the library

#### Framework

We are using the pdm backend to build the package.

#### Version

The versioning is linked with the tag. To build a local version, you can add a tag, build version and then delete tag.  
The tag is expected to have the following form: `v0.1.2` and support alpha and beta version `v0.1.2a0`.  

```console
    git tag  v0.2.0b1
    uv build # --> dist/mechaphlowers-0.2.0b1-...whl
    git tag -d v0.2.0b1
```

The variable PACKAGE_BUILD_TEST can be used to add the `.devX`version.

```console
    git tag  v0.2.0b1
    export PACKAGE_BUILD_TEST=3
    uv build # --> dist/mechaphlowers-0.2.0b1.dev3-...whl
    git tag -d v0.2.0b1
```


#### Build

In order to build the library (wheel and tar.gz archive):

```console
    uv build
```

You can check the build option to control the output folder or the desired output file types.  


### How to serve the documentation

You can build and serve the documentation using or `make docs`:

```console
    uv run mkdocs serve -a localhost:8001
```

### Testing in a browser via pyodide

You may test your pyodide package using pyodide console in a browser.

#### Download pyodide

Download a version of Pyodide from the [releases page](https://github.com/pyodide/pyodide/releases/), extract it and serve it with a web server:

    wget https://github.com/pyodide/pyodide/releases/download/0.25.0/pyodide-0.25.0.tar.bz2
    tar -xvf pyodide-0.25.0.tar.bz2
    cd pyodide
    python3 -m http.server

Pyodide console is then available at http://localhost:8000/console.html

#### Test in pyodide console

Copy needed wheels to pyodide folder.
Then, in pyodide console:

    import micropip
    # load your wheel
    await micropip.install("http://localhost:8000/<wheel_name>.whl", keep_going=True)

