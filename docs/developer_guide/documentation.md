# Mechaphlowers Documentation

## Building documentation 

### Serving documnentation

cf. README

### Building documentation

If you need to export, you can use the build command from mkdocs.
See [https://www.mkdocs.org/](https://www.mkdocs.org/) for more informations.

## Adding documentation

Website structure is defined in the `nav` section of the mkdocs.yaml.

Use the docs/docs folder to organize where the documentation can be added :

- Getting started for installation and first steps
- User guide for user documentation
- Developer guide for technical documentation
- _static is for images, figures
- docstring is for automatic docstring generation (see below)
- javascript is only for addons or extra plugins of mkddocs


## Dosctring generation

You have to complete the folder docstring with the subrepositories you want. At the end put a markdown with anchor `::: package.module.submodule` where submodule is the module you want to auto generate.

!!! important

    The file has to be placed inside subrepositories reflecting the code folders architecture. The name of the .md file containing the anchor needs to correspond to the .py file to document.

## Opening plotly figures

Plotly figures are not rendered in the documentation but available in html. A json version is available in the same folder in order to modify the figure if needed.  
To open plotly figures, you need to use the following code :

```python
from plotly import io as pio

# open it
fig = pio.read_json("./docs/user_guide/assets/how_span_engine_work.json")

# modify it
fig._data[0]['marker']['size'] = 2
fig._data[1]['marker']['size'] = 2
fig._data[2]['marker']['size'] = 2
fig._data[3]['marker']['size'] = 2

# verify
fig.show()

# save it
fig.write_html("./docs/user_guide/assets/how_span_engine_work.html")
fig.write_json("./docs/user_guide/assets/how_span_engine_work.json")
```

## Configuration

The configuration can be found in mkdocs.yaml :

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