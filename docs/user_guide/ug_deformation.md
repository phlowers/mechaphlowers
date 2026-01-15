# Deformation

The deformation class implements the cable deformation models described in [Cable modeling](ug_cable_model.md#physics-based-cable-modeling).

This class is called in two situations:

- When computing $L_{ref}$, which is done only one time
- When solving the new state after giving new wind pressure, ice thickness and temperature in `find_parameter_solver.py`. In this case, the solver calls `IDeformation` many times in order to find the solution

In the former case, the computation is made without wind, ice, and class attribute `current_temperature` is equal to `sagging_temperature` (usually 15°C). That is why `current_temperature` equals to `sagging_temperature` by default when creating the class.

In the latter case, the wind and ice are taken into account through `load_coefficient`, the new temperature is given directly by changing the attribute `current_temperature` of `IDeformation`

```python

data_container = factory_data_container(section_array, cable_array, weather_array)
data_container.sagging_temperature = 15

# create Deformation with initial state, current_temperature is set to sagging_temperature
deformation_model = DeformationRte(
	**data_container.__dict__,
	tension_mean=tension_mean,
	cable_length=cable_length,
)
deformation_model.current_temperature
# 15


# input new temperature after a state change
deformation_model.current_temperature = 25
```