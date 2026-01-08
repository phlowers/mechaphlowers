# Thermal Engine User Guide

## Overview

The `ThermalEngine` is a wrapper around the thermohl library that enables comprehensive cable thermal modeling. It allows you to compute cable temperatures under various environmental and operational conditions, supporting both steady-state and transient thermal analysis.

## Key Concepts

### What is the ThermalEngine?

The ThermalEngine computes the temperature distribution in power transmission cables based on:

- Cable properties (resistance, thermal conductivity, dimensions)
- Operational parameters (electrical current)
- Environmental conditions (ambient temperature, wind speed, winf angle, solar radiation)
- Geographic location and time (latitude, longitude, altitude, month, day, hour)

### Thermal Calculations Supported

1. **Steady-State Temperature**: Calculates cable temperature when conditions are constant
2. **Steady-State Intensity**: Calculates maximum allowable current for a target temperature
3. **Transient Temperature**: Calculates temperature variations over time with changing conditions

## Installation and Basic Setup

### Initialization

```python
from mechaphlowers.core.models.cable.thermal import ThermalEngine
from mechaphlowers.data.catalog.catalog import sample_cable_catalog

# Create a thermal engine instance
thermal_engine = ThermalEngine()
```

### Setting Input Parameters

Use the `set()` method to configure all input parameters:

```python
import numpy as np

# Get a cable from the catalog
cable_array = sample_cable_catalog.get_as_object(["ASTER600"])

# Set thermal engine parameters
thermal_engine.set(
    cable_array=cable_array,
    latitude=45.0,
    longitude=0.0,
    altitude=0.0,
    azimuth=0.0,  # Cable direction (degrees from north)
    month=3,
    day=21,
    hour=12,
    intensity=500.0,  # Current in Amperes
    ambient_temp=15.0,  # Ambient temperature in Celsius
    wind_speed=10.0,  # Wind speed in m/s
    wind_angle=90.0,  # Wind direction angle (degrees from north)
    solar_irradiance=None,  # Optional: solar radiation (W/m²)
)
```

## Parameters Reference

| Parameter | Type | Unit | Description |
|-----------|------|------|-------------|
| `cable_array` | CableArray | - | Cable properties (diameter, resistance, thermal conductivity, etc.) |
| `latitude` | float/array | degrees | Geographic latitude |
| `longitude` | float/array | degrees | Geographic longitude |
| `altitude` | float/array | meters | Altitude above sea level |
| `azimuth` | float/array | degrees | Cable direction (0°=North, 90°=East, 180°=South, 270°=West) |
| `month` | int/array | 1-12 | Month of the year |
| `day` | int/array | 1-31 | Day of the month |
| `hour` | int/array | 0-23 | Hour of the day |
| `intensity` | float/array | Amperes | Electrical current through the cable |
| `ambient_temp` | float/array | °C | Ambient air temperature |
| `wind_speed` | float/array | m/s | Wind speed magnitude |
| `wind_angle` | float/array | degrees | Wind direction (0°=North, 90°=East) |
| `solar_irradiance` | float/array | W/m² | Solar radiation (optional, auto-calculated if None) |

## Features and Usage

### 1. Scalar Inputs (Single Location/Condition)

Calculate thermal conditions for a single set of conditions:

```python
thermal_engine.set(
    cable_array=cable_array,
    latitude=45.0,
    longitude=0.0,
    altitude=0.0,
    azimuth=0.0,
    month=3,
    day=21,
    hour=12,
    intensity=500.0,
    ambient_temp=15.0,
    wind_speed=10.0,
    wind_angle=90.0,
)

# Compute steady-state temperature
result = thermal_engine.steady_temperature()
print(result.data)
```

### 2. Array Inputs (Multiple Locations/Conditions)

Process multiple conditions simultaneously. **All arrays must have the same length**:

```python
# Three different geographical locations or time periods
thermal_engine.set(
    cable_array=cable_array,
    latitude=np.array([45.0, 45.0, 45.0]),
    longitude=np.array([0.0, 1.0, 2.0]),
    altitude=np.array([0.0, 100.0, 200.0]),
    azimuth=np.array([0.0, 0.0, 90.0]),
    month=np.array([3, 3, 3]),
    day=np.array([21, 21, 21]),
    hour=np.array([12, 12, 12]),
    intensity=np.array([500.0, 600.0, 700.0]),
    ambient_temp=np.array([15.0, 18.0, 20.0]),
    wind_speed=np.array([10.0, 5.0, 15.0]),
    wind_angle=np.array([90.0, 90.0, 90.0]),
)

result = thermal_engine.steady_temperature()
print(f"Number of results: {len(result)}")
print(result.data)
```

### 3. Steady-State Temperature Calculation

Calculate the cable temperature at equilibrium for given operating conditions:

```python
# Basic usage
result = thermal_engine.steady_temperature()
print("Steady-state results:")
print(result.data)

# Override intensity for calculation
result = thermal_engine.steady_temperature(intensity=np.array([800.0, 900.0, 1000.0]))
print("Results with different intensity:")
print(result.data)
```

**Output columns:**

- `t_avg`: Average cable temperature
- `t_surf`: Surface temperature
- `t_core`: Core temperature (highest temperature in the conductor)

### 4. Steady-State Intensity Calculation

Calculate the maximum allowable current to maintain a target temperature:

```python
# Set target temperature (default is 65°C)
thermal_engine.target_temperature = 70.0

# Calculate maximum intensity
result = thermal_engine.steady_intensity()
print("Maximum intensity for 70°C target:")
print(result.data)

# Override target temperature for calculation
result = thermal_engine.steady_intensity(target_temperature=75.0)
print("Maximum intensity for 75°C target:")
print(result.data)
```

**Use case:** Determine the maximum current a line can carry while maintaining safe temperature limits.

### 5. Transient Temperature Calculation

Simulate temperature evolution over time with time-varying conditions:

```python
# Calculate temperature variations over 10 time steps
result = thermal_engine.transient_temperature()
print("Transient results:")
print(result.data)
```

**Output format:**

- `time`: Time step
- `id`: Cable/condition ID
- `t_avg`: Average temperature at that time
- `t_surf`: Surface temperature at that time
- `t_core`: Core temperature at that time

**Customize the forecast:**

```python
from mechaphlowers.core.models.cable.thermal import ThermalForecastArray

# Define custom time series
forecast = ThermalForecastArray()
forecast.time = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
forecast.wind_speed = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])  # Decreasing wind
forecast.ambient_temp = np.linspace(15, 25, 10)  # Rising temperature

result = thermal_engine.transient_temperature(forecast_control=forecast)
print(result.data)
```

### 6. Dynamic Parameter Updates

Modify parameters after initialization without resetting the entire engine:

```python
# Initial setup
thermal_engine.set(
    cable_array=cable_array,
    latitude=45.0,
    longitude=0.0,
    altitude=0.0,
    azimuth=0.0,
    month=3,
    day=21,
    hour=12,
    intensity=500.0,
    ambient_temp=15.0,
    wind_speed=10.0,
    wind_angle=90.0,
)

# Update specific parameters
thermal_engine.dict_input["I"] = 600.0  # Change intensity
thermal_engine.dict_input["Ta"] = 20.0  # Change ambient temperature
thermal_engine.load()  # Reload with new parameters

# Compute new results
result = thermal_engine.steady_temperature()
print("Updated results:", result.data)
```

### 7. Wind-Cable Angle Calculation

Automatically compute the angle between wind direction and cable orientation:

```python
# This property is calculated based on azimuth and wind_angle
wind_cable_angle = thermal_engine.wind_cable_angle
print(f"Wind-cable angle: {wind_cable_angle}°")
```

This is important because wind cooling effectiveness depends on the wind direction relative to the cable.

### 8. String Representations

Inspect engine configuration using string representations:

```python
# Brief string representation
print(str(thermal_engine))
# Output: power_model=rte, heateq=3t

# Detailed representation
print(repr(thermal_engine))
# Output: <ThermalEngine(power_model=rte, heateq=3t)>

# Check engine size
print(f"Number of conditions: {len(thermal_engine)}")
```

## Complete Examples

### Example 1: Temperature Monitoring for Multiple Lines

Monitor cable temperature across three transmission lines at different locations:

```python
import numpy as np
from mechaphlowers.core.models.cable.thermal import ThermalEngine
from mechaphlowers.data.catalog.catalog import sample_cable_catalog

# Load cable
cable = sample_cable_catalog.get_as_object(["ASTER600"])

# Create engine
engine = ThermalEngine()

# Set up three lines at different locations
engine.set(
    cable_array=cable,
    latitude=np.array([45.2, 44.8, 46.1]),
    longitude=np.array([0.5, 1.5, 2.0]),
    altitude=np.array([100, 150, 200]),
    azimuth=np.array([0, 45, 90]),
    month=np.array([6, 6, 6]),  # June
    day=np.array([21, 21, 21]),  # Summer solstice
    hour=np.array([14, 14, 14]),  # Afternoon peak
    intensity=np.array([800, 900, 750]),
    ambient_temp=np.array([25, 28, 23]),
    wind_speed=np.array([3, 5, 2]),
    wind_angle=np.array([90, 90, 45]),
)

# Calculate steady-state temperatures
temps = engine.steady_temperature()
print("Cable Temperatures:")
print(temps.data[['t_core', 't_surf', 't_avg']])

# Check which cables exceed 70°C
hot_cables = temps.data[temps.data['t_core'] > 70]
print(f"\nCables exceeding 70°C: {len(hot_cables)}")
```

### Example 2: Finding Safe Current Limits

Determine the maximum current each line can carry while keeping temperatures below 75°C:

```python
engine.target_temperature = 75.0  # Target: 75°C max

# Calculate maximum intensity for each line
max_currents = engine.steady_intensity()
print("Maximum Safe Currents (for 75°C target):")
print(max_currents.data[['I']])  # I column contains maximum intensity

# Compare with current operating intensity
current_intensity = np.array([800, 900, 750])
safe_margin = max_currents.data['I'].values - current_intensity
print(f"\nSafety margins: {safe_margin} A")
```

### Example 3: Temperature Evolution During Load Change

Simulate how cable temperature changes as load increases over time:

```python
# Initial conditions
engine.set(
    cable_array=cable,
    latitude=45.0,
    longitude=0.0,
    altitude=0.0,
    azimuth=0.0,
    month=6,
    day=21,
    hour=12,
    intensity=500.0,
    ambient_temp=25.0,
    wind_speed=5.0,
    wind_angle=90.0,
)

# Calculate transient response
transient = engine.transient_temperature()
temps_df = transient.data

# Plot temperature evolution
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(temps_df['time'], temps_df['t_core'], label='Core Temperature', marker='o')
plt.plot(temps_df['time'], temps_df['t_surf'], label='Surface Temperature', marker='s')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.title('Cable Temperature Evolution Over Time')
plt.show()
```

### Example 4: Seasonal Analysis

Compare cable performance across different seasons:

```python
seasons = {
    'Winter': (1, 21, 12),    # January 21, noon
    'Spring': (4, 21, 12),    # April 21, noon
    'Summer': (7, 21, 12),    # July 21, noon
    'Fall': (10, 21, 12),     # October 21, noon
}

seasonal_temps = {}

for season_name, (month, day, hour) in seasons.items():
    engine.set(
        cable_array=cable,
        latitude=45.0,
        longitude=0.0,
        altitude=0.0,
        azimuth=0.0,
        month=month,
        day=day,
        hour=hour,
        intensity=800.0,
        ambient_temp=10.0 + 15.0 * np.sin((month - 1) * np.pi / 6),  # Temperature variation
        wind_speed=10.0,
        wind_angle=90.0,
    )
    
    result = engine.steady_temperature()
    seasonal_temps[season_name] = result.data['t_core'].values[0]

print("Seasonal Peak Temperatures (at 800A):")
for season, temp in seasonal_temps.items():
    print(f"  {season}: {temp:.1f}°C")
```

## Error Handling

The ThermalEngine validates input consistency:

```python
import numpy as np

try:
    # This will raise an error - array lengths don't match
    engine.set(
        cable_array=cable,
        latitude=np.array([45.0, 45.0, 45.0]),  # 3 values
        longitude=np.array([0.0, 1.0]),          # 2 values - mismatch!
        altitude=0.0,
        azimuth=0.0,
        month=3,
        day=21,
        hour=12,
        intensity=500.0,
        ambient_temp=15.0,
        wind_speed=10.0,
        wind_angle=90.0,
    )
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: All array inputs must have the same length.
```

## Technical Notes

### Default Parameters

- **Target Temperature**: 65°C (can be modified via `thermal_engine.target_temperature`)
- **Power Model**: RTE (Réseau de Transport d'Électricité)
- **Heat Equation**: 3-Temperature model (core, surface, average)
- **Solar Irradiance**: Automatically calculated if not provided

### Temperature Output Interpretation

- **t_core**: Highest temperature in the conductor (most critical)
- **t_surf**: Temperature at cable surface
- **t_avg**: Average temperature across the conductor

Typically: `t_core > t_avg > t_surf`

### Wind Angle Conventions

- 0° = Wind from North
- 90° = Wind from East
- 180° = Wind from South
- 270° = Wind from West

### Cable Azimuth Conventions

- 0° = Cable oriented North-South (wind blows perpendicular = maximum cooling)
- 90° = Cable oriented East-West
- The most effective cooling occurs when wind is perpendicular to the cable (90° wind-cable angle)

## Limitations and Future Enhancements

Currently implemented:
- Steady-state temperature and intensity calculations
- Transient temperature analysis
- Multi-location/condition analysis
- Automatic solar irradiance calculation

Not yet implemented:
- Normal wind mode
- Custom thermal parameters per cable
- Integration with cable catalogs for direct lookup
- Visualization tools

See the code comments for planned improvements.

## See Also

- [Cable Arrays Documentation](../user_guide/ug_arrays.md)
- [Cable Catalog Guide](../user_guide/ug_catalog.md)
- [Balance Engine Documentation](../user_guide/ug_balance_engine.md)
