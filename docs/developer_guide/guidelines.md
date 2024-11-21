# Guidelines 

## Library structure

- an __api__ module for all the main objects used by the user
- an __entities__ module for all the classes dedicated to data
- an __adapter__ module houses all the functions and interfaces needed to guarantee decoupling
- an __core__ module hosts the library scientific core and is divided in two parts:
    - the __models__ part
    - the __solvers__ part
- a __data__ module hosts all the helpers generating objects as sample cables for example
- a __graphics__ module for the code relating to vizualization.


## Object responsability structure

Some guidelines: 

- Isolate the class with user-interaction
- Expose interfaces for bypassing user-interaction and allows custom use with extranal services for example.
- Propose input/output interfaces to offer customization (units, data parameters)
- Propose a modeler/solver separation
- Separate models with solvers from the rest of the code to facilitate the addition of other models or solvers in the future.
- Whenever possible, use classes that depend on table-based data structures, such as dataframes, to reduce mapping steps and enhance performance.


## Framework

One of the challenges of the package is to use it offline, running in a web browser. To achieve this, the choice has been made to use [Pyodide](https://github.com/pyodide/pyodide).

The use in the pyodide framework constraint the project to use the versions proposed by pyodide available [here](https://pyodide.org/en/stable/usage/packages-in-pyodide.html).

The stack’s purpose is to remain small. It is based almost on the scipy stack:
- numpy, scipy, autograd for scientific core
- pandas for data-user interaction
- plotly for user interaction figures

If some mathematical functions are available in scikit-learn, adding it in the stack is a possibility.


## Performance

In order to ensure good performances and minimize consumption, the library have to be compliant with python's just in time compilers. The current choice is [numba](https://numba.pydata.org/) it is not fixed: for example the python 3.13 version offers a python-base just in time compiler.

To monitor the library consumption, the possibility to set metrics with [codecarbon](https://github.com/mlco2/codecarbon) will be studied.


## Installation options

In the same idea of reduction of the packages dependencies, the project targets to offer different installation options function of usage (for example user-local, offline, server...)