import os

import numpy as np

from orion.config.my_config import *
from orion.map.map_search.search_voxel import VLMapSearch
from orion.map.occupancy import OccupancyMapping

data_dir = "data/experiments/predict_4ok3usBNeis_B1_U1"

map_querier = VLMapSearch(
    load_sparse_map_path=os.path.join(data_dir, "lseg_vlmap/sparse_vxl_map.npz"),
)

occu_map = np.load(os.path.join(data_dir, "occupancy_map.npy"))
navigatable_mask = np.load(os.path.join(data_dir, "gt_navigable_mask.npy"))

navigatable_mask_crop = navigatable_mask[
    map_querier.zmin : map_querier.zmax + 1, map_querier.xmin : map_querier.xmax + 1
]
wall_mask = occu_map == OccupancyMapping.WALL
wall_mask_crop = wall_mask[
    map_querier.zmin : map_querier.zmax + 1, map_querier.xmin : map_querier.xmax + 1
]


predict_map, query_labels = map_querier.query(["fridge"])
predict_map_crop = predict_map[
    map_querier.zmin : map_querier.zmax + 1, map_querier.xmin : map_querier.xmax + 1
]

for tgt_name in query_labels:
    if tgt_name in ["other", "floor", "wall"]:
        continue
    map_querier.plan(
        tgt_name,
        query_labels,
        predict_map_crop,
        navigatable_mask_crop,
        wall_mask_crop,
        show=True,
    )
    # red dot is the ceter of the target object
    # yellow dot is the viewpoint
