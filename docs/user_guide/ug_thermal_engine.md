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

Use the `set()` method to configure all input parameters. **All inputs must be numpy arrays**:

```python
import numpy as np

# Get a cable from the catalog
cable_array = sample_cable_catalog.get_as_object(["ASTER600"])

# Set thermal engine parameters (all inputs as numpy arrays)
thermal_engine.set(
    cable_array=cable_array,
    latitude=np.array([45.0]),
    longitude=np.array([0.0]),
    altitude=np.array([0.0]),
    azimuth=np.array([0.0]),  # Cable direction (degrees from north)
    month=np.array([3]),
    day=np.array([21]),
    hour=np.array([12]),
    intensity=np.array([500.0]),  # Current in Amperes
    ambient_temp=np.array([15.0]),  # Ambient temperature in Celsius
    wind_speed=np.array([10.0]),  # Wind speed in m/s
    wind_angle=np.array([90.0]),  # Wind direction angle (degrees from north)
    solar_irradiance=None,  # Optional: solar radiation (W/m²)
)
```

## Parameters Reference

| Parameter | Type | Unit | Description |
|-----------|------|------|-------------|
| `cable_array` | CableArray | - | Cable properties (diameter, resistance, thermal conductivity, etc.) |
| `latitude` | np.ndarray | degrees | Geographic latitude |
| `longitude` | np.ndarray | degrees | Geographic longitude |
| `altitude` | np.ndarray | meters | Altitude above sea level |
| `azimuth` | np.ndarray | degrees | Cable direction (0°=North, 90°=East, 180°=South, 270°=West) |
| `month` | np.ndarray | 1-12 | Month of the year |
| `day` | np.ndarray | 1-31 | Day of the month |
| `hour` | np.ndarray | 0-23 | Hour of the day |
| `intensity` | np.ndarray | Amperes | Electrical current through the cable |
| `ambient_temp` | np.ndarray | °C | Ambient air temperature |
| `wind_speed` | np.ndarray | m/s | Wind speed magnitude |
| `wind_angle` | np.ndarray | degrees | Wind direction (0°=North, 90°=East) |
| `solar_irradiance` | np.ndarray \| None | W/m² | Solar radiation (optional, auto-calculated if None) |

## Features and Usage

### 1. Single Location/Condition

Calculate thermal conditions for a single set of conditions (using single-element arrays):

```python
thermal_engine.set(
    cable_array=cable_array,
    latitude=np.array([45.0]),
    longitude=np.array([0.0]),
    altitude=np.array([0.0]),
    azimuth=np.array([0.0]),
    month=np.array([3]),
    day=np.array([21]),
    hour=np.array([12]),
    intensity=np.array([500.0]),
    ambient_temp=np.array([15.0]),
    wind_speed=np.array([10.0]),
    wind_angle=np.array([90.0]),
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

Modify parameters after initialization without resetting the entire engine. **Note: All values must be numpy arrays with the same length**:

```python
# Initial setup
thermal_engine.set(
    cable_array=cable_array,
    latitude=np.array([45.0, 46.0]),
    longitude=np.array([0.0, 1.0]),
    altitude=np.array([0.0, 100.0]),
    azimuth=np.array([0.0, 45.0]),
    month=np.array([3, 3]),
    day=np.array([21, 21]),
    hour=np.array([12, 14]),
    intensity=np.array([500.0, 600.0]),
    ambient_temp=np.array([15.0, 18.0]),
    wind_speed=np.array([10.0, 8.0]),
    wind_angle=np.array([90.0, 90.0]),
)

# Update specific parameters (must be numpy arrays of same length)
thermal_engine.dict_input["I"] = np.array([700.0, 800.0])  # Change intensity
thermal_engine.dict_input["Ta"] = np.array([20.0, 22.0])  # Change ambient temperature
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

