# Distances computation

## Definition

In the context of mechaphlowers, distances between a point and a curve do NOT refer to the measurement of the shortest path between a point and a line (span or section) in 3D space.

In order to compute distances, we first define a plane containing the point. This plane is defined by the point and a normal vector.  
We take the normal vector as the direction of 2 foot points "sea level" to keep a vertical Z-axis.  

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


### Several obstacles at once

[`DistanceEngine.distances`][mechaphlowers.core.geometry.distances.DistanceEngine.distances] is the low-level method computing the distances of several obstacles together. It works on plain arrays in the section frame and does not modify the state of the engine. [`PositionEngine.get_distances`][mechaphlowers.core.geometry.position_engine.PositionEngine.get_distances] gathers the obstacle data and calls it (see [Distances to obstacles](#distances-to-obstacles)).

```python
from mechaphlowers.core.geometry.distances import DistanceEngine

engine = DistanceEngine()
obstacles = engine.distances(
    coords_dict={"road": road_coords},  # {name: array (V, 3)}, in point_index order
    spans_dict={"road": 0},  # {name: span index}
    curves=curves,  # array (S, M, 3): points of each span
    normals=normals,  # array (S, 3): unit horizontal normal of each span
    step=1.0,
    refine=True,
    xtol=1e-3,
    geometry_types={"road": "polyline"},  # optional, missing obstacles are polylines
)
obstacles["road"].minimum.distance_3d
```

`DistanceEngine.min_distances` takes the same arguments and returns only the `minimum` of each polyline obstacle.

### Plot engine

The `PlotEngine` class contains also method to compute and visualize distances.  
This more complete object gives access to `point_distance` method to compute the distance of an obstacle point and visualize the setup upon the plot of a section for example.  

```python
from mechaphlowers.plotting.plot import PlotEngine

balance_engine = ...  # BalanceEngine object with computed balance (use data.catalog.sample_section_factory for sample data)
plt_engine = PlotEngine(balance_engine)
plt_engine.position_engine.add_obstacle(name="obs_0", span_index=0, coords=np.array([[10.0, 5.0, 2.0]]))
fig = figure_factory()
distance_result = plt_engine.point_distance(obstacle_name="obs_0", point_index=0, fig=fig)
# ...get a distance result object with the distance and closest point coordinates
            
fig.show()
```

--8<-- "docs/user_guide/assets/distance_plotengine_example.html"

### Distances to obstacles

[`get_distances`][mechaphlowers.core.geometry.position_engine.PositionEngine.get_distances] computes the distances between obstacles and the span they belong to. It is available on `PositionEngine` and on `SectionStudy`. It takes the obstacle names to compute (a string, a list of strings, or `None` for all obstacles) and returns a dict mapping each obstacle name to an [`ObstacleDistances`][mechaphlowers.core.geometry.distances.ObstacleDistances]:

- `points`: a dict mapping each obstacle vertex (its position in `point_index` order) to its `DistanceResult`: `point_base` is the obstacle point and `point_target` the target point on the cable. Points whose distance plane does not intersect the span are absent;
- `minimum`: the [`ObstacleDistanceResult`][mechaphlowers.core.geometry.distances.ObstacleDistanceResult] of the closest obstacle location, or `None` if no point intersects the span;
- `geometry_type` and `span_index`: the obstacle geometry type and the index of its span.

How `minimum` is searched depends on the obstacle `geometry_type` (see [Discrete points or polyline](ug_obstacles.md#discrete-points-or-polyline)):

- `"points"` (default): the points are discrete, the minimum is the closest point. `step` and `refine` have no effect. `segment_index` is the position of the closest point and `segment_parameter` is `0`;
- `"polyline"`: the points are joined in `point_index` order, and the minimum is searched along the polyline, so it can lie between two points.

For polylines, the algorithm is:

1. Each segment is sampled every `step` meters (vertices included) and the plane distance is computed at each sample.
2. If `refine=True` (default), a golden-section search refines the best sample within its neighbouring interval, up to `xtol` meters along the segment. This search is run simultaneously for all obstacles.

All obstacles and spans are computed in batched numpy operations, without Python loops over points.

Points or samples whose distance plane does not intersect the span (for instance beyond the supports) are skipped with a `NoIntersectionPlaneWarning`, which reports the number of skipped samples per obstacle. `step` and `xtol` must be strictly positive (`ValueError`), and an unknown obstacle name raises `KeyError`.

```python
study = SectionStudy(...)  # with a solved balance
study.position_engine.add_obstacle(
    name="road",
    span_index=0,
    coords=np.array([[100, -30, 0], [400, 30, 0]]),
    geometry_type="polyline",
)

road = study.get_distances("road", step=1.0)["road"]
road.points[0].distance_3d  # distance of the first point
road.points[0].point_target  # target point on the cable (absolute frame)

result = road.minimum
result.distance_3d  # minimum distance
result.closest_obstacle_point  # closest point on the obstacle (absolute frame)
result.closest_span_point  # closest point on the cable (absolute frame)
result.segment_index, result.segment_parameter  # location along the obstacle polyline
result.distance_result  # DistanceResult at the closest location

# All obstacles, or a selection
results = study.get_distances()
results = study.get_distances(["road", "tree"])

# Minimum only, without the points
result = study.obstacle_min_distance("road", step=1.0)

# With plotting of the closest location
fig = go.Figure()
study.plot_engine.obstacle_min_distance("road", fig=fig)
```

## Algorithms explained

### Vectorized plane distance 

Problem:

- $N$ obstacle points lie on $S$ spans
- each span has $M$ points
- 3D means 3 components

We want a vectorized function that computes (implemented as [`batch_plane_distances`](../docstring/core/geometry/distance.md#mechaphlowers.core.geometry.distances.batch_plane_distances) in `mechaphlowers.core.geometry.distances`):

- the targets $T \in \mathbb{R}^{N \times 3}$ (points of the cable that are in the span axis plane containing the obstacles)
- the distances $d \in \mathbb{R}^{N}$

#### Inputs:

- normal vectors to the $S$ span axis planes: $n \in \mathbb{R}^{S \times 3}$
- obstacle points: $p \in \mathbb{R}^{N \times 3}$
- span ids of each obstacle point: $s \in \{0, \dots, S-1\}^{N}$
- $S$ curves of $M$ points in a 3D matrix: $C \in \mathbb{R}^{S \times M \times 3}$

#### Algorithm

1. Project the points on the normals

    $$
    \pi^{obs}_k = \vec{p_k} \cdot \vec{n_{s_k}} \qquad \text{shape } (N,)
    $$

    !!! note

        $\vec{n_{s_k}}$ (`n[span_ids]`) is the repeated $\vec{n}$ to map from $S$ to $N$ size.

    $$
    \pi^{curve}_{i,j} = \vec{C_{i,j}} \cdot \vec{n_i} \qquad \text{shape } (S, M)
    $$

2. Get signed distances

    $$
    D_{k,j} = \pi^{obs}_k - \pi^{curve}_{s_k,\, j} \qquad \text{shape } (N, M)
    $$

    $D$ is thus the signed distance matrix with $N$ rows of distances between each obstacle plane and the points of the corresponding curve.

3. Sort $\lvert D \rvert$ and find $c_0$, $c_1$ the closest points of the curve at indices $i_0$, $i_1$, with $\vec{c_0}, \vec{c_1} \in \mathbb{R}^{N \times 3}$

4. Exclude degenerate cases

    - verify crossing (catch the sign change of $D$)
    - compute the vectors $\vec{c_0 c_1}$
    - $\vec{c_0 c_1} \cdot \vec{n_{s_k}}$ gives the direction
    - exclude $\lvert \vec{c_0 c_1} \cdot \vec{n_{s_k}} \rvert \approx 0$ to avoid segments parallel to the plane

5. Find the best point $\vec{c(t)}$ that lies in the plane by minimizing the projected distance between the obstacle and the point on the segment $[\vec{c_0}, \vec{c_1}]$

    Linear approximation: let's find $t$ on the segment:

    $$
    \vec{c(t)} = \vec{c_0} + t \, \vec{c_0 c_1}
    $$

    We have to solve:

    $$
    \vec{n_{s_k}} \cdot \vec{c(t)} = \pi^{obs}_k
    \iff \vec{n_{s_k}} \cdot \vec{c_0} + t \, \left( \vec{n_{s_k}} \cdot \vec{c_0 c_1} \right) = \pi^{obs}_k
    $$

    hence:

    $$
    t = \frac{\pi^{obs}_k - \vec{n_{s_k}} \cdot \vec{c_0}}{\vec{n_{s_k}} \cdot \vec{c_0 c_1}}
    $$

    (the distance to the plane of our guess is the same as the projected obstacle, which means the plane contains the obstacle point)

6. Filter the valid guesses by clipping $t$ to the interval $[0, 1]$





