# Geography — GPS Coordinate Utilities

In order to build a `SectionArray`, mechaphlowers needs **span length** and **line angle** data. These are the primary geometric inputs for any computation.

In practice, support positions may be known as GPS coordinates (latitude / longitude), or as Lambert coordinates. Therefore, one may want to generate GPS coordinates/Lambert coordinates from the section geometry or, conversely, recover the section geometry from GPS/Lambert surveys.



| Function | Direction |
|---|---|
| `get_gps_from_arrays` | Section geometry ➜ GPS coordinates |
| `get_dist_and_angles_from_gps` | GPS coordinates ➜ Section geometry |

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
