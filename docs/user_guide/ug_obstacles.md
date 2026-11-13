# Obstacles

## Adding obstacles

### ObstacleArray.\_\_init\_\_()

In order to add obstacles to mechaphlowers, you can use the `ObstacleArray` object.

This class allows to store obstacles data. You can add several obstacles, several points per obstacles, and a variable number of points per obstacle.

Note that this data needs to be in relation to a SectionArray to make sense (especially the coordinates).


```python
input_data = {
	"name": ["obs_0", "obs_1", "obs_0", "obs_2", "obs_1", "obs_1"],
	"point_index": [0, 1, 1, 0, 2, 0],
	"span_index": [0, 1, 0, 1, 1, 1],
	"x": [
		100.0,
		200.0,
		100.0,
		200.0,
		300.0,
		350.0,
	],
	"y": [0.0, 10.0, 0.0, 0.0, 10.0, 0.0],
	"z": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
	"object_type": [
		"ground",
		"ground",
		"ground",
		"ground",
		"ground",
		"ground",
	],
}
obs_array = ObstacleArray(pd.DataFrame(input_data))
obs_array.data

```

### ObstacleArray.add_obstacle()


Alternatively, you can use the the `ObstacleArray.add_obstacle()`, in order to add obstacles one by one.

That way, you can

```python

input_data = {
	"name": ["obs_0", "obs_0"],
	"point_index": [0, 1],
	"span_index": [0, 0],
	"x": [
		100.0,
		200.0,
	],
	"y": [0.0, 10.0],
	"z": [0.0, 0.0],
	"object_type": [
		"ground",
		"ground",
	],
}
obstacle_array = ObstacleArray(pd.DataFrame(input_data))
obstacle_array.add_obstacle(
	name="obs_1",
	span_index=1,
	coords=np.array([[100, 0, 0], [200, 0, 10], [300, 10, 0], ]),
	support_reference='left',
)
obstacle_array.add_obstacle(
	name="obs_2",
	span_index=1,
	coords=np.array([[50, 0, 0]]),
	support_reference='right',
	span_length=np.array([500, 400, np.nan]),
    )
```


### Discrete points or polyline

The optional `geometry_type` column (or `geometry_type` argument of `add_obstacle()`) tells how the points of a multipoint obstacle are interpreted for distance computation:

- `"points"` (default): discrete points;
- `"polyline"`: points joined in `point_index` order, for instance for a road or a power line crossing.

If the column is absent from the DataFrame, all obstacles are `"points"`. All points of an obstacle must have the same `geometry_type`, otherwise a `ValueError` is raised.

```python
input_data = {
	"name": ["road", "road", "pylon", "pylon"],
	"point_index": [0, 1, 0, 1],
	"span_index": [0, 0, 0, 0],
	"x": [100.0, 400.0, 150.0, 150.0],
	"y": [-30.0, 30.0, 0.0, 5.0],
	"z": [0.0, 0.0, 0.0, 0.0],
	"object_type": ["ground", "ground", "ground", "ground"],
	"geometry_type": ["polyline", "polyline", "points", "points"],
}
obs_array = ObstacleArray(pd.DataFrame(input_data))
```

The same flag is available with `add_obstacle()`, for instance on a `SectionStudy`:

```python
study.position_engine.add_obstacle(
	name="road",
	span_index=0,
	coords=np.array([[100, -30, 0], [400, 30, 0]]),
	geometry_type="polyline",
)
```

See [Distances](ug_distances.md#distances-to-obstacles) for the distance computation.


## Plotting obstacles

You can plot obstacles using PlotEngine:

```python
import plotly.graph_objects as go

plt_engine = PlotEngine(balance_engine)

plt_engine.add_obstacles(obs_array)
fig = go.Figure()
plt_engine.preview_line3d(fig)
fig.show()

```

That way, the obstacles will show up along the span when calling `preview_line3d()`

You can get the obstacle coordinates using `PlotEngine.obstacles_dict()` or `PlotEngine.get_obstacles_points()`

```python
plt_engine.obstacles_dict()
# {'obs_0': array([
# 	[100.,   0.,   0.],
# 	[200.,  10.,   0.]
# 	]),
# 	'obs_1': array([
# 		[598.76883406,  15.6434465 ,   0.        ],
# 		[699.10201277,  41.16377641...97.87084683,  56.80722292,  50.        ]
# 		]), 
# 	'obs_2': array([[694.40897882,  11.5331262 ,   0.        ]])
# 	}
plt_engine.get_obstacles_points()

# np.array(
#         [
#             [np.nan, np.nan, np.nan],
#             [100.0, 0.0, 0.0],
#             [200.0, 10.0, 0.0],
#             [np.nan, np.nan, np.nan],
#             [598.76883406, 15.6434465, 0.0],
#             [699.10201277, 41.16377641, 0.0],
#             [797.87084683, 56.80722292, 50.0],
#             [np.nan, np.nan, np.nan],
#             [694.40897882, 11.5331262, 0.0],
#         ]
#     )
```

--8<-- "docs/user_guide/assets/obstacle_plot_example.html"


## Additional points

In addition to obstacles, you can also add generic "additional points". These points use the exact same underlying structure (`ObstacleArray`) and validation, but they are stored separately so they do not mix with physical obstacles with geometric points or analyses.

You can add and manage additional points directly using the `PositionEngine`:

```python
# Assuming `study` is an initialized SectionStudy
study = SectionStudy(...)

# Add additional points
study.position_engine.add_additional_point(
    name="pt_0",
    span_index=0,
    coords=np.array([[10.0, 0.0, -5.0]])
)

# You can also use an array built in advance
# study.position_engine.add_additional_points_array(my_obstacle_array)
```

Once added, you can extract their coordinates. Additional points are not part of the obstacle distance computation (`get_distances`):

```python
# Get coordinates as a single continuous absolute coordinate array
additional_coords = study.position_engine.get_additional_points()

# Or get them from the global points topology
group_points = study.position_engine.get_group_points()
group_points_dict = group_points.additional_points_dict()
# {'pt_0': array([[...]])}
```