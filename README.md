# Library

Physical calculation package for the mechanics and geometry of overhead power lines.

## Set up development environment

You need python 3.11. You may have to install it manually (e.g. with pyenv).

Then you may create a virtualenv, install dependencies and activate the env:

    # create virtual env (if needed) and install dependencies (including dev dependencies)
    poetry install
    poetry shell  # activate virtual env

Tip: if using VSCode/VSCodium, configure it to use your virtual env's interpreter.

## How to format or lint code

Once dev dependencies are installed, you may format and lint python files like this:

    poetry run poe format
    poetry run poe lint

Use following command if you only want to check if files are correctly formatted:

    poetry run poe check-format

You may automatically fix some linting errors:

    poetry run poe lint-fix

Tip: if using VSCode/VSCodium, you may also use Ruff extension.

## How to check typing

In order to check type hints consistency, you may run:

    poetry run poe typing

## How to test

### On the command line:

    poetry run poe test

### In VSCode:

Configure VSCode to use your virtual env's interpreter.
Open the Testing tab and configure tests using pytest.
Click to run tests.

## All in one

You may run every check mentioned above with just one command:

    poetry run poe checks

## Exporting the library

In order to build the library (wheel and tar.gz archive):

    poetry build

## How to serve the documentation

    poetry install --with docs  # install documentation related dependencies
    poetry run poe doc

# Testing in a browser via pyodide

You may test your pyodide package using pyodide console in a browser.

## Download pyodide

Download a version of Pyodide from the [releases page](https://github.com/pyodide/pyodide/releases/), extract it and serve it with a web server:

    wget https://github.com/pyodide/pyodide/releases/download/0.25.0/pyodide-0.25.0.tar.bz2
    tar -xvf pyodide-0.25.0.tar.bz2
    cd pyodide
    python3 -m http.server

Pyodide console is then available at http://localhost:8000/console.html

## Test in pyodide console

Copy needed wheels to pyodide folder.
Then, in pyodide console:

    import micropip
    # load your wheel
    await micropip.install("http://localhost:8000/<wheel_name>.whl", keep_going=True)

