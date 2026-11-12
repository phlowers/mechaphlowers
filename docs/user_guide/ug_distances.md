# Distances computation

## Definition

In the context of mechaphlowers, distances between a point and a curve do NOT refer to the measurement of the shortest path between a point and a line (span or section) in 3D space.

In order to compute distances, we first define a plane containing the point. This plane is defined by the point and a normal vector.  
We take the normal vector as the direction of 2 foot points "sea level" to keep a vertical Z-axis.  

### Computation steps

The process of computing distances is then as follows:

1. Define the plane based on the point and the normal vector.
2. Compute the intersection of the plane with the curve (span or section).
3. Compute the distance from the point to the intersection point on the curve in the plane.
4. Compute the projection of the distance vector onto the plane basis vectors to get the distance components in the plane.

### Intersection of plane and curve

The intersection of the plane and the curve is computed using the `intersection_curve_plane` function, which takes the curve points, plane point, and plane normal as inputs and returns the intersection point on the curve.  
The function calculates the 2 nearest points on the curve. Then it uses a linear interpolation to find the intersection point on the curve.

## How to use it

### Distance engine

The simplest way to compute distances is to use the `DistanceEngine` class, which encapsulates all the steps of the distance computation process.
You can create an instance of the `DistanceEngine` class by providing the axis start and end points, add a curve and get the distance with an input point.

```python
from mechaphlowers.core.geometry.distances import DistanceEngine

# Create a distance engine with the axis start and end points
distance_engine = DistanceEngine()
# Add a curve (span or section) to the distance engine
distance_engine.add_curves(curve_points)
# Add a frame
distance_engine.add_span_frame(x_axis_start=np.array([0, 0, 0]), x_axis_end=np.array([1, 0, 0]))
# Compute the distance from a point to the curve
distance_result = distance_engine.plane_distance(point_base)
```

You can also visualize your setup and the distance result using the `plot_distance_engine` function from `mechaphlowers.plotting.plot_distances`:

```python
from mechaphlowers.plotting.plot_distances import plot_distance_engine

fig = plot_distance_engine(distance_engine, distance_result=distance_result)
fig.show()
```

### Distance result

The `DistanceResult` class is used to store the results of the distance computation. It contains the following attributes:

- `point_base`: The input point from which the distance is computed.
- `point_target`: The intersection point on the curve.
- `u_plane`: The u-axis of the plane.
- `v_plane`: The v-axis of the plane.
- `distance_3d`: The 3D distance from the point to the curve.
- `distance_projection_u`: The distance from the point to the curve projected onto the u-axis
- `distance_projection_v`: The distance from the point to the curve projected onto the v-axis


### Plot engine

The `PlotEngine` class contains also method to compute and visualize distances.  
This more complete object gives access to `point_distance` method to compute distances and visualize the setup upon the plot of a section for example.  

```python
from mechaphlowers.plotting.plot import PlotEngine

balance_engine = ...  # BalanceEngine object with computed balance (use data.catalog.sample_section_factory for sample data)
plt_engine = PlotEngine(balance_engine)
point = np.array([10.0, 5.0, 2.0])  # Absolute coordinates of the point to analyze
fig = figure_factory()
distance_result = plt_engine.point_distance(span_index=0, point=point)
# ...get a distance result object with the distance and closest point coordinates
            
fig.show()
```

--8<-- "docs/user_guide/assets/distance_plotengine_example.html"

### Minimum distance to an obstacle

An obstacle made of several points is considered as a polyline: its points are joined in `point_index` order.
`obstacle_min_distance` returns the minimum distance between this polyline and the span the obstacle belongs to, using the plane distance described above.

The algorithm is:

1. Each obstacle segment is sampled every `step` meters (vertices included) and the plane distance is computed at each sample.
2. If `refine=True` (default), a golden-section search is run around the best sample, so the minimum can lie between two samples. This search is run simultaneously for all obstacles.

All obstacles and spans are computed in batched numpy operations, without Python loops over points.

Samples whose distance plane does not intersect the span (for instance beyond the supports) are skipped with a `NoIntersectionPlaneWarning`. If no sample intersects the span, `None` is returned.

```python
study = SectionStudy(...)  # with a solved balance
study.position_engine.add_obstacle(
    name="road",
    span_index=0,
    coords=np.array([[100, -30, 0], [400, 30, 0]]),
)

result = study.obstacle_min_distance("road", step=1.0)
result.distance_3d  # minimum distance
result.closest_obstacle_point  # closest point on the obstacle (absolute frame)
result.closest_span_point  # closest point on the cable (absolute frame)
result.segment_index, result.segment_parameter  # location along the obstacle polyline
result.distance_result  # DistanceResult at the closest location

# All obstacles at once
results = study.get_min_distances_from_obstacles()

# With plotting
fig = go.Figure()
study.plot_engine.obstacle_min_distance("road", fig=fig)
```
