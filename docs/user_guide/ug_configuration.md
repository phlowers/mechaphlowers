# Configuration

In order to centralize the definition of some important parameters, we have created a dedicated module called `config.py` folder. This file contains only some configuration parameters that are set globally.  
Those parameters are accessible through the `options` module available at mechaphlowers level (see example below).

The following code snippet shows how to import and use these parameters:
```python
import mechaphlowers as mph

# Computations before plotting
balance_engine = mph.BalanceEngine(cable_array, section_array)
balance_engine.solve_adjustment()
balance_engine.solve_change_state()

# use options to set the graphics resolution and marker size
mph.options.graphics.marker_size = 10
mph.options.graphics.resolution = 20

# plot then without changing locally those parameters
plot_engine = mph.PlotEngine(balance_engine)
fig = go.Figure()
plot_engine.preview_line3d(fig)
fig.show()
```