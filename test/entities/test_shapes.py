import numpy as np

from mechaphlowers.entities.shapes import SupportShape


def test_shapes__with_x_values():
    shape_values = np.array(
        [
            [0, 0, 18.5],
            [3, -3, 14.5],
            [-3, -3, 14.5],
            [0, -9, 14.5],
        ]
    )
    number_of_nonzeros_on_x_arms = len(shape_values[shape_values[:, 0] != 0.0])

    pyl_shape = SupportShape(
        name="pyl",
        xyz_arms=shape_values,
        set_number=np.array([22, 46, 47, 55]),
    )

    assert len(pyl_shape.arms_points) == (2 + 1) * len(shape_values)
    assert (
        len(pyl_shape.arms_set_points)
        == (2 + 1) * number_of_nonzeros_on_x_arms
    )

    np.testing.assert_almost_equal(
        pyl_shape.support_points,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 18.5],
                [np.nan, np.nan, np.nan],
                [0.0, 0.0, 18.5],
                [0.0, 0.0, 18.5],
                [np.nan, np.nan, np.nan],
                [0.0, 0.0, 14.5],
                [0.0, -3.0, 14.5],
                [np.nan, np.nan, np.nan],
                [0.0, 0.0, 14.5],
                [0.0, -3.0, 14.5],
                [np.nan, np.nan, np.nan],
                [0.0, 0.0, 14.5],
                [0.0, -9.0, 14.5],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [0.0, -3.0, 14.5],
                [3.0, -3.0, 14.5],
                [np.nan, np.nan, np.nan],
                [0.0, -3.0, 14.5],
                [-3.0, -3.0, 14.5],
                [np.nan, np.nan, np.nan],
            ]
        ),
    )

    np.testing.assert_almost_equal(pyl_shape.labels_points, shape_values)

    assert True


def test_shapes__without_x_values():
    shape_values = np.array(
        [
            [0, 0, 18.5],
            [0, -3, 14.5],
            [0, 6, 16.5],
            [0, -9, 12.5],
        ]
    )
    number_of_nonzeros_on_x_arms = len(shape_values[shape_values[:, 0] != 0.0])

    pyl_shape = SupportShape(
        name="pyl",
        xyz_arms=shape_values,
        set_number=np.array([22, 46, 47, 55]),
    )

    assert len(pyl_shape.arms_points) == (2 + 1) * len(shape_values)
    assert (
        len(pyl_shape.arms_set_points)
        == (2 + 1) * number_of_nonzeros_on_x_arms
    )

    np.testing.assert_almost_equal(
        pyl_shape.support_points,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 18.5],
                [np.nan, np.nan, np.nan],
                [0.0, 0.0, 18.5],
                [0.0, 0.0, 18.5],
                [np.nan, np.nan, np.nan],
                [0.0, 0.0, 14.5],
                [0.0, -3.0, 14.5],
                [np.nan, np.nan, np.nan],
                [0.0, 0.0, 16.5],
                [0.0, 6.0, 16.5],
                [np.nan, np.nan, np.nan],
                [0.0, 0.0, 12.5],
                [0.0, -9.0, 12.5],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
            ]
        ),
    )

    np.testing.assert_almost_equal(pyl_shape.labels_points, shape_values)


def test_shapes__arms_length_logic():
    shape_values = np.array(
        [
            [0, 0, 18.5],
            [4, -3, 14.5],
            [0, 6, 16.5],
            [0, -9, 12.5],
        ]
    )
    pyl_shape = SupportShape(
        name="pyl",
        xyz_arms=shape_values,
        set_number=np.array([22, 46, 47, 55]),
    )

    np.testing.assert_almost_equal(
        pyl_shape.arms_length, np.array([0, 5, 6, 9])
    )
