# Configuration

In order to centralize the definition of some important parameters, we have created a dedicated module called `config.py` folder. This file contains only some configuration parameters that are set globally.  
Those parameters are accessible through the `options` module available at mechaphlowers level (see example below).

The following code snippet shows how to import and use these parameters:
```python
import mechaphlowers as mph

frame = mph.SectionDataFrame(section)

# use options to set the graphics resolution and marker size
mph.options.graphics_marker_size = 10
mph.options.graphics_resolution = 20

# plot then without changing locally those parameters
fig = go.Figure()
frame.plot.line3d(fig)
```