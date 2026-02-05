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


## Plotting obstacles

You can plot obstacles using PlotEngine:

```python
import plotly.graph_objects as go

plt_engine = PlotEngine.builder_from_balance_engine(balance_engine)

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