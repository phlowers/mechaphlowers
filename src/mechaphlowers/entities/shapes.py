from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


class SupportShape:
    """Support shape class enables to store a support complete "set" representation.
    The support is then represented by a set of arms, each arm being represented by x,y,z coordinate.
    Assumption for the representation:

        - The support is centered with origin at the support ground.
        - Trunk is a vertical bar
        - base arms are on Y coordinate only, from the trunk
        - set point are on X coordinate only, from the edge of the base arms
    The class supposes the support centered with origin at the support ground.
    However, the ground origin can be moved by user.

    Usages: generated with support catalog, and use plotting.plot_support_shape to visualize.
    """

    def __init__(
        self,
        name: str,
        xyz_arms: np.ndarray,
        set_number: np.ndarray,
        ground_point: np.ndarray = np.array([0, 0, 0]),
        arms_length: np.ndarray | None = None,
    ):
        """Support Shape

        Args:
            name (str): support name
            xyz_arms (np.ndarray): (n, 3) array with x,y,z set point coordinates in the support frame (n is the number of arms)
            set_number (np.ndarray): (n,) array with the set numbers
            ground_point (np.ndarray, optional): ground point of the support. Defaults to np.array([0, 0, 0]).
            arms_length (np.ndarray | None, optional): (n,) array with the arms lengths. Defaults to None means the amrs are the norm of x,y values
        """
        self.name = name
        self.ground_point = ground_point
        self.xyz_arms = xyz_arms
        if arms_length is None:
            self.arms_length = np.linalg.norm(xyz_arms[:, 0:2], axis=1)
        else:
            self.arms_length = arms_length
        self.set_number = set_number

    @property
    def trunk_points(self) -> np.ndarray:
        """trunk_points

        Returns:
            np.ndarray: (2, 3) array with the points of the trunk of the support in points format
        """
        return np.array([self.ground_point, [0, 0, max(self.xyz_arms[:, 2])]])

    @property
    def arms_points(self) -> np.ndarray:
        """arms_points

        Returns:
            np.ndarray: (3*n, 3) array with the points of the arms of the support in points format
        """

        point_2 = self.xyz_arms.copy()
        point_2[:, 0] = 0.0
        point_1 = point_2.copy()
        point_1[:, 1] = 0.0
        mix_points = np.hstack([point_1, point_2, point_2 * np.nan])
        return np.reshape(mix_points, (len(self.xyz_arms) * 3, 3))

    @property
    def arms_set_points(self) -> np.ndarray:
        """arms_set_points

        Support shape can have X coordinates different from 0, which means several set points for the same arm. This property
        returns the points of the arms of the support in points format.

        Returns:
            np.ndarray: (3*n, 3) with n the number on __nonzero__ values on the X coordinates of the arms of the support. If n=0 returns array([np.nan, np.nan, np.nan])
        """
        mask_for_nonzero_set_points = self.xyz_arms[:, 0]
        point_2 = self.xyz_arms[
            np.abs(mask_for_nonzero_set_points) > 1e-6
        ].copy()
        if len(point_2) == 0:
            return np.full((0, 3), np.nan)
        point_1 = point_2.copy()
        point_1[:, 0] = 0.0
        mix_points = np.hstack([point_1, point_2, point_2 * np.nan])
        return np.reshape(mix_points, (len(point_2) * 3, 3))

    @property
    def labels_points(self) -> np.ndarray:
        """labels_points

        Returns:
            np.ndarray: (n,) array with set numbers of the arms of the support
        """
        points = self.xyz_arms.copy()
        return points

    @property
    def support_points(self) -> np.ndarray:
        """support_points

        Returns:
            np.ndarray: ( (n+1)*3, 3) array with the points of the trunk of the support in points format
        """
        return np.vstack(
            [
                self.trunk_points,
                np.zeros(3) * np.nan,
                self.arms_points,
                np.zeros(3) * np.nan,
                self.arms_set_points,
            ]
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
                xyz_arms = df.loc[n, ['X', 'Y', 'Z']].to_numpy()
                set_number = df.loc[n, ['set_number']].to_numpy()
                support_shape_list.append(
                    SupportShape(
                        name=n,
                        xyz_arms=xyz_arms,
                        set_number=set_number,
                        ground_point=np.array([0, 0, 0]),
                    )
                )
        else:
            raise IndexError(
                "The asked key is missing from catalog index. Verify the key or the catalog name ?"
            )
        return support_shape_list
