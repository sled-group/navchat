from typing import List, Optional

import numpy as np

from orion import logger
from orion.config.my_config import MapConfig


class MapSearch:
    def __init__(
        self,
        load_sparse_map_path: str,
        map_index_bound: Optional[List[int]] = None,
        mapcfg: MapConfig = MapConfig(),
    ):
        self.mapcfg = mapcfg

        self._load_sparse_map(load_sparse_map_path)
        self.update_index(map_index_bound)

    def _load_sparse_map(self, load_path):
        # Load the indices and values
        sparse_map = np.load(load_path)
        self.indices = sparse_map["indices"]
        self.feat_values = sparse_map["feat_values"]
        self.vxl_count = sparse_map["count_values"]
        self.rgb_values = sparse_map["rgb_values"]
        self.gt_values = sparse_map["gt_values"]
        self._3dshape = (
            self.mapcfg.num_grid,
            self.mapcfg.num_grid,
            self.mapcfg.num_vxl_height,
        )
        logger.info(f"load_path: {load_path}")

    def update_index(self, map_index_bound: Optional[List[int]] = None):
        "save search time when using this to crop the vxlmap"
        z_indices, x_indices, y_indices = (
            self.indices[:, 0],
            self.indices[:, 1],
            self.indices[:, 2],
        )
        no_map_mask = np.zeros(shape=(self.mapcfg.num_grid, self.mapcfg.num_grid))
        no_map_mask[z_indices, x_indices] = 1
        self.no_map_mask = np.logical_not(no_map_mask)

        if map_index_bound is not None:
            self.xmin, self.xmax, self.zmin, self.zmax = map_index_bound
        else:
            self.xmin = np.min(x_indices)
            self.xmax = np.max(x_indices)
            self.zmin = np.min(z_indices)
            self.zmax = np.max(z_indices)
            self.ymin = np.min(y_indices)
            self.ymax = np.max(y_indices)

        # logger.info(
        #     f"map_index_bound: {self.xmin}, {self.xmax}, {self.zmin}, {self.zmax}"
        # )
        self.no_map_mask_crop = self.no_map_mask[
            self.zmin : self.zmax + 1, self.xmin : self.xmax + 1
        ]

    @staticmethod
    def get_BEV_map(indices, values, map_shape):
        assert indices.shape[-1] == 3
        assert indices.shape[0] == values.shape[0]
        assert values.shape[-1] == map_shape[-1]
        assert len(map_shape) == 4

        rev_indices = indices[
            ::-1
        ]  # reverse to get the largest y value for each (z, x) easily
        rev_z, rev_x, rev_y = rev_indices[:, 0], rev_indices[:, 1], rev_indices[:, 2]

        # Create a unique identifier for each (z, x) pair
        rev_unique_zx, rev_paired_y = np.unique(
            np.column_stack((rev_z, rev_x)), axis=0, return_index=True
        )

        # Find the maximum 'y' value for each unique (z, x) pair
        max_y_values = rev_y[rev_paired_y]

        bev_indices_sparse = np.column_stack(
            (rev_unique_zx, max_y_values)
        )  # [z, x, top_y]

        paired_y = len(indices) - rev_paired_y - 1

        bev_feat_sparse = values[paired_y]  # [z, x, feat_dim]
        bev_map = np.zeros(
            shape=(map_shape[0], map_shape[1], map_shape[3])
        )  # [num_z, num_x, feat_dim]
        bev_map[bev_indices_sparse[:, 0], bev_indices_sparse[:, 1], :] = bev_feat_sparse

        return bev_map
