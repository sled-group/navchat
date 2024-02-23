import argparse
import os
import time

import torch
from orion.map.map_build.build_voxel import VoxelMapBuilder, OfflineDataLoader
from orion.config.my_config import MapConfig, SCENE_ID_FLOOR_SET
from orion.utils.file_load import get_floor_set_str
from tqdm import tqdm

from orion.config.my_config import *
from orion.utils import visulization as vis
from orion.map.map_search.search_voxel import VLMapSearch
from orion.perception.extractor.concept_fusion_extractor import ConceptFusionExtractor
from orion.perception.extractor.lseg_extractor import LSegExtractor


def build_vlmap_one_scene(root_dir, feature_type):
    map_builder = VoxelMapBuilder(
        save_dir=os.path.join(root_dir, f"{feature_type}_vlmap"), 
        extractor_type = "lseg" if feature_type == "lseg" else "conceptfusion",
        extractor = LSegExtractor() if feature_type == "lseg" else ConceptFusionExtractor(),
        accelerate_mapping=True
    )

    dataloader = OfflineDataLoader(
        data_dir=os.path.join(root_dir, "recordings"),
        mapcfg=MapConfig(),
    )

    for idx in tqdm(range(len(dataloader))):
        obs = dataloader[idx]
        map_builder.build(obs)

    map_builder._save()

    # just make sure realease GPU memory
    del map_builder.extractor
    del map_builder.vxlmap
    del map_builder


def draw_vlmap_one_scene(root_dir, feature_type):
    # Draw topdown rgb
    map_querier = VLMapSearch(
        load_sparse_map_path=os.path.join(
            root_dir, f"{feature_type}_vlmap", "sparse_vxl_map.npz")
    )
    mapshape = map_querier._3dshape
    mapshape = (mapshape[0], mapshape[1], mapshape[2], 3)
    topdown_rgb = map_querier.get_BEV_map(
        indices=map_querier.indices, 
        values=map_querier.rgb_values,
        map_shape=mapshape
    )
    vis.plot_uint8img_with_plt(
        topdown_rgb,
        "topdown_rgb",
        crop=True,
        save=True,
        save_path=os.path.join(
            root_dir, f"{feature_type}_vlmap", "topdown_rgb.png"),
    )

    # Test query list
    predict_map, query_labels = map_querier.query(VLMAP_QUERY_LIST_COMMON)
    nomap_mask_crop = map_querier.no_map_mask_crop
    predict_map_crop = predict_map[
        map_querier.zmin : map_querier.zmax + 1, 
        map_querier.xmin : map_querier.xmax + 1
    ]
    vis.plot_BEV_semantic_map(
        predict_map_crop,
        nomap_mask_crop,
        labels=query_labels,
        save=True,
        save_path=os.path.join(
            root_dir,  f"{feature_type}_vlmap", "topdown_vlmap.png"),
    )

    del map_querier
    
    

def main(scene_id, floor, feature_type):
    data_dir = f"data/experiments/predict_{scene_id}_{get_floor_set_str(floor)}"
    build_vlmap_one_scene(data_dir, feature_type)
    torch.cuda.empty_cache()
    time.sleep(5)

    draw_vlmap_one_scene(data_dir, feature_type)
    torch.cuda.empty_cache()
    time.sleep(5)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--scene_id",
        type=str,
        default="4ok3usBNeis",
        help="scene id, either 'all' or a specific scene id in SCENE_ID_FLOOR_SET",
    )
    argparser.add_argument(
        "--feature_type",
        choices=["lseg", "conceptfusion"],
        default="lseg",
        help="feature type, either 'lseg' or 'conceptfusion'",
    )
    args = argparser.parse_args()
    
    scene_dic = {item[0]: item for item in SCENE_ID_FLOOR_SET}    
    if args.scene_id == "all":
        for scene_id, floor in SCENE_ID_FLOOR_SET:
            main(scene_id, floor, args.feature_type)
    else:
        assert args.scene_id in scene_dic
        scene_id, floor = scene_dic[args.scene_id]
        main(scene_id, floor, args.feature_type)