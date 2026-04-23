# Geography — GPS Coordinate Utilities

In order to build a `SectionArray`, mechaphlowers needs **span length** and **line angle** data. These are the primary geometric inputs for any computation.

In practice, support positions may be known as GPS coordinates (latitude / longitude), or as Lambert 93 coordinates. Therefore, one may want to generate GPS/Lambert coordinates from the section geometry or, conversely, recover the section geometry from GPS/Lambert surveys.

## Overview

| API | Direction | Description |
|---|---|---|
| `SectionArray.set_starting_gps` / `get_gps` | Section geometry ➜ GPS | High-level: set origin on `SectionArray`, retrieve GPS for all pylons |
| `SectionArray.set_starting_lambert93` / `get_lambert93` | Section geometry ➜ Lambert 93 | High-level: set Lambert 93 origin, retrieve Lambert 93 for all pylons |
| `GeoLocator` | Section geometry ➜ GPS / Lambert 93 | Low-level computation class used internally by `SectionArray` |
| `get_gps_from_arrays` | Section geometry ➜ GPS coordinates | Standalone function |
| `get_dist_and_angles_from_gps` | GPS coordinates ➜ Section geometry | Standalone function |
| `get_dist_and_angles_from_lambert` | Lambert 93 ➜ Section geometry | Standalone function |

---

## Using GeoLocator via SectionArray

The recommended way to work with GPS and Lambert 93 coordinates on a section is through `SectionArray`, which owns a `GeoLocator` instance. You set the starting point once, then call `get_gps()` or `get_lambert93()` to compute coordinates for all pylons on demand.

### Set starting point from GPS, retrieve GPS coordinates

```python
import numpy as np
import pandas as pd
from mechaphlowers.entities.arrays import SectionArray

section_array = SectionArray(
    pd.DataFrame({
        "name": ["P1", "P2", "P3", "P4", "P5"],
        "suspension": [False, True, True, True, False],
        "conductor_attachment_altitude": [20.0, 5.0, 10.0, 0.0, 0.0],
        "crossarm_length": [10.0, 12.1, 10.0, 10.1, 5.0],
        "line_angle": [0.0, 10.0, 15.0, 20.0, 30.0],
        "insulator_length": [1.0, 4.0, 3.2, 1.0, 1.0],
        "span_length": [300.0, 400.0, 500.0, 600.0, float("nan")],
        "insulator_mass": [1000.0, 500.0, 500.0, 500.0, 1000.0],
    })
)
section_array.add_units({"line_angle": "deg"})

# Set the GPS origin of the first pylon and the azimuth of the first span
section_array.set_starting_gps(
    latitude_0=48.8566,   # decimal degrees
    longitude_0=2.3522,
    azimuth_0=0.0,        # 0 = North, 90 = West (anti-clockwise)
)

latitudes, longitudes = section_array.get_gps()
print("Latitudes :", latitudes)
print("Longitudes:", longitudes)
```

### Set starting point from Lambert 93, retrieve Lambert 93 coordinates

```python
# Set the origin from Lambert 93 easting/northing
section_array.set_starting_lambert93(
    easting=652544.0,
    northing=6861023.0,
    azimuth_0=0.0,
)

easting_arr, northing_arr = section_array.get_lambert93()
print("Easting :", easting_arr)
print("Northing:", northing_arr)
```

### Error: starting point not set

Accessing coordinates before setting a starting point raises `GpsNoDataAvailable`:

```python
from mechaphlowers.entities.errors import GpsNoDataAvailable

new_section = SectionArray(...)  # no set_starting_gps called
try:
    new_section.get_gps()
except GpsNoDataAvailable as e:
    print(e)  # GPS data is not available. Call set_starting_gps() or set_starting_lambert93() first.
```

See [Error Classes](../docstring/entities/errors.md) for details.

---

## Low-level: GeoLocator class

`GeoLocator` is the computation object that `SectionArray` delegates to. It can also be used directly if you want to manage the starting point independently of a `SectionArray`.

```python
import numpy as np
from mechaphlowers.entities.geography import GeoLocator

geolocator = GeoLocator()
geolocator.set_starting_gps(48.8566, 2.3522, azimuth_0=0.0)

line_angles = np.array([0.0, 10.0, 15.0, 20.0, 0.0])  # degrees
span_lengths = np.array([300.0, 400.0, 500.0, 600.0, float("nan")])

latitudes, longitudes = geolocator.get_gps(line_angles, span_lengths)
easting, northing = geolocator.get_lambert93(line_angles, span_lengths)
```

`GeoLocator` stores **only** the starting point and azimuth — no computed arrays are cached. Every call to `get_gps()` / `get_lambert93()` recomputes from the stored starting point.

See [Entities Geography API](../docstring/entities/geography.md) for the full class reference.

---

## From span lengths and angles to GPS 

### Purpose

Starting from a **single GPS origin** and an **initial azimuth** (direction of the first span), this function iteratively builds the GPS position of every support in the section using:

- the **line angles**,
- the **span lengths**.

Internally, the function converts all angles to radians, computes cumulative bearings, and applies the **reverse Haversine formula** at each step.

### Example

```python
import numpy as np
from mechaphlowers.entities.geography import get_gps_from_arrays

# 5 supports, 4 spans
line_angles = np.array([0.0, 10.0, 15.0, 20.0, 30.0])   # degrees
span_lengths = np.array([300.0, 400.0, 500.0, 600.0, np.nan])  # meters

latitudes, longitudes = get_gps_from_arrays(
    start_lat=48.8566,   # Paris latitude
    start_lon=2.3522,    # Paris longitude
    azimuth=0.0,         # first span heads North
    line_angles_degrees=line_angles,
    span_length=span_lengths,
)

print("Latitudes :", latitudes)
print("Longitudes:", longitudes)
# Latitudes : [48.8566     48.85929597 48.86182017 48.86472027 48.86712227]
# Longitudes: [ 2.3522      2.3522      2.35285838  2.35504186  2.35908232]
```

Each returned array has one element per support (length = number of supports).

---

## From GPS to span lengths and angles

### Purpose

Given **arrays of GPS coordinates** (one per support), this function recovers:

- the **distances** between consecutive supports (in meters), and
- the **relative line angles** at each support (in degrees).

This is the "inverse" direction: from a real-world GPS survey back to the section geometry needed by mechaphlowers.

### Example

```python
import numpy as np
from mechaphlowers.entities.geography import get_dist_and_angles_from_gps

# 5 supports with known GPS positions
latitudes = np.degrees(np.array(
    [0.852708, 0.8527550884, 0.852816919, 0.8528880459, 0.8529546364]
))
longitudes = np.degrees(np.array(
    [0.041053, 0.041053, 0.0410695724, 0.0411199932, 0.0412212353]
))

distances, angles = get_dist_and_angles_from_gps(latitudes, longitudes)

print("Distances (m):", distances)
print("Angles (deg) :", angles)
# Distances (m): [300. 400. 500. 600.  nan]
# Angles (deg) : [ 0. 10. 15. 20.  0.]
```

---
