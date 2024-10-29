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

# Using mechaphlowers

    from mechaphlowers import welcome
    welcome()

# Testing in a browser via pyodide

You may test your pyodide package using pyodide console in a browser.

## Clone pyodide

Clone pyodide project:

    git clone https://github.com/pyodide/pyodide
    cd pyodide
    git checkout 0.25.0

## Start a server

In pyodide folder:

    python3 -m http.server --directory dist

Pyodide console is then available at http://localhost:8000/console.html

## Copy needed wheels in pyodide's dist folder

Copy your wheel to pyodide's dist folder.

## Test in pyodide console

In the console:

    import micropip
    # load your wheel
    await micropip.install("http://localhost:8000/<wheel_name>.whl", keep_going=True)
