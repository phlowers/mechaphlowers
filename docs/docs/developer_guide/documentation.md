# Mechaphlowers Documentation

## Building documentation 

### Serving documnentation

From inside docs folder : 
```{bash}
cd ./docs
mkdocs serve -a 0.0.0.0:8001
```

### Building documentation

If you need to export, you can use the build command from mkdocs.
See [https://www.mkdocs.org/](https://www.mkdocs.org/) for more informations.

## Adding documentation

Website structure is defined in the `nav` section of the mkdocs.yaml.

Use the docs/docs folder to organize where the documentation can be added :

- Getting started for installation and first steps
- User guide for user documentation
- Developer guide for technique documentation
- _static is for images, figures
- docstring is for automatic docstring generation (see below)
- javascript is only for addons or extra plugins of mkddocs


## Dosctring generation

You have to complete the folder docstring with the subrepositories you want. At the end put a markdown with anchor `::: package.module.submodule` where submodule is the module you want to auto generate.


## Configuration

The configuration can be found in docs/mkdocs.yaml :

- plugins for docstrings generation, jupyter, search, and offline mode
- markdown extension
    - snippets for including markdown located outside of the docs folder
    - arithmatex for mathjax support. Warning it relies on a small js file in the javascripts folder.

## Stack choice

Mechaphlowers use the mkdocs stack for documentation.
This choice is based on :

- simplicity
- have to handle auto docstring generation
- have to handle math latex equations
- have to include plotly figures in html


Mkdocs need some configuration to reach those needs :

- mkdocs - _principal package_
- mkdocstrings - _extension to generate docstring_
- mkdocs-material - _material theme used_
- mkdocs-jupyter - _extension to include jupyter notebooks_ (not mandatory)

In the future, the package [mike](https://github.com/jimporter/mike) could be used to handle [multiple version of documentation](https://squidfunk.github.io/mkdocs-material/setup/setting-up-versioning/)

## CI

Some help from mkdocs to configure CI on github can be found [here](https://squidfunk.github.io/mkdocs-material/publishing-your-site/)