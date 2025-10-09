from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


class SupportShape:
    """Support shape class enables to store a support complete "set" representation.
    The support is then represented by a set of arms, each arm being represented by the altitude and arm length.
    The class supposes the support centered with origin at the support ground.
    However, the ground origin can be moved by user.

    Usages: from support catalog, for plot_support_shape to visualize.
    """

    def __init__(
        self,
        name: str,
        yz_arms: np.ndarray,
        set_number: np.ndarray,
        ground_point: np.ndarray = np.array([0, 0, 0]),
    ):
        """Support Shape

        Args:
            name (str): support name
            yz_arms (np.ndarray): (n, 2) array with arm length in first coordinate and altitude after (n is the number of arms)
            set_number (np.ndarray): (n,) array with the set numbers
            ground_point (np.ndarray, optional): ground point of the support. Defaults to np.array([0, 0, 0]).
        """
        self.name = name
        self.ground_point = ground_point
        self.arms = yz_arms
        self.set_number = set_number

    @property
    def trunk_points(self) -> np.ndarray:
        """trunk_points

        Returns:
            np.ndarray: (2, 3) array with the points of the trunk of the support in points format
        """
        return np.array([self.ground_point, [0, 0, max(self.arms[:, 1])]])

    @property
    def arms_points(self) -> np.ndarray:
        """arms_points

        Returns:
            np.ndarray: (3*n, 3) array with the points of the arms of the support in points format
        """
        point_2 = np.vstack([self.arms.T[0] * 0, self.arms.T]).T
        point_1 = np.vstack([self.arms.T[0:2] * 0, self.arms.T[1]]).T
        mix_points = np.hstack([point_1, point_2, point_2 * np.nan])
        return np.reshape(mix_points, (len(self.arms) * 3, 3))

    @property
    def labels_points(self) -> np.ndarray:
        """labels_points

        Returns:
            np.ndarray: (n,) array with set numbers of the arms of the support
        """
        points = np.vstack([self.arms.T[0] * 0, self.arms.T]).T
        return points

    @property
    def support_points(self) -> np.ndarray:
        """support_points

        Returns:
            np.ndarray: ( (n+1)*3, 3) array with the points of the trunk of the support in points format
        """
        return np.vstack(
            [self.trunk_points, np.zeros(3) * np.nan, self.arms_points]
        )

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> List[SupportShape]:
        """from_dataframe allows to generate a list of SupportShape object from a dataframe

        Args:
            df (pd.DataFrame): _description_

        Raises:
            IndexError: if the dataframe has no index

        Returns:
            List[SupportShape]: List of SupportShape objects
        """

        name = df.index.unique().tolist()

        if len(name) >= 1:
            support_shape_list = []
            for n in name:
                arms = df.loc[n, ['arm_length', 'arm_altitude']].to_numpy()
                set_number = df.loc[n, ['set_number']].to_numpy()
                support_shape_list.append(
                    SupportShape(n, arms, set_number, np.array([0, 0, 0]))
                )
        else:
            raise IndexError(
                "The asked key is missing from catalog index. Verify the key or the catalog name ?"
            )
        return support_shape_list
