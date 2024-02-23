import os
import numpy as np

from orion import logger
from orion.utils.geometry import CoordinateTransform
from orion.config.my_config import *
from orion.utils import file_load as load_utils
from orion.map.voxel import VoxelMapping
from orion.map.voxel_sparse import VoxelMappingSparse
from orion.abstract.perception import ExtractorModule
from orion.perception.extractor.lseg_extractor import LSegExtractor
from orion.utils.geometry import PinholeCameraModel as pinhole
from orion.abstract.interfaces import Observations


class OfflineDataLoader:
    def __init__(self, data_dir, mapcfg: MapConfig = MapConfig()):
        rgb_dir = os.path.join(data_dir, "rgb")
        depth_dir = os.path.join(data_dir, "depth")
        pose_dir = os.path.join(data_dir, "pose")
        semantic_dir = os.path.join(data_dir, "semantic")

        rgb_list = sorted(
            os.listdir(rgb_dir), key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        depth_list = sorted(
            os.listdir(depth_dir), key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        pose_list = sorted(
            os.listdir(pose_dir), key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        semantic_list = sorted(
            os.listdir(semantic_dir), key=lambda x: int(x.split("_")[-1].split(".")[0])
        )

        rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
        depth_list = [os.path.join(depth_dir, x) for x in depth_list]
        pose_list = [os.path.join(pose_dir, x) for x in pose_list]
        semantic_list = [os.path.join(semantic_dir, x) for x in semantic_list]

        self._data = list(zip(rgb_list, depth_list, semantic_list, pose_list))

        self.obj2cls_dic, _ = load_utils.load_obj2cls_dict(
            os.path.join(data_dir, "obj2cls_dict.txt")
        )
        self.mapcfg = mapcfg

    def __getitem__(self, idx):
        rgb_path, depth_path, semantic_path, pose_path = self._data[idx]
        rgb = load_utils.load_image(rgb_path)
        depth = load_utils.load_depth(depth_path)
        semantic = load_utils.load_semantic(semantic_path, self.obj2cls_dic)
        simpose = load_utils.load_pose(pose_path)
        if idx == 0:
            self.transform_fn = CoordinateTransform(
                num_grd=self.mapcfg.num_grid,
                cell_size=self.mapcfg.cell_size,
                init_agtpose=simpose,
            )
        cam_pose = self.transform_fn.get_relative_campose(simpose)
        return Observations(
            rgb=rgb,  # [h, w, 3]  uint8
            depth=depth,  # [h, w] float32
            semantic=semantic,  # [h, w]  int
            rel_cam_pose=cam_pose,  # [4, 4]
        )

    def __len__(self):
        return len(self._data)


class VoxelMapBuilder:
    """This is working in another process"""

    def __init__(
        self,
        save_dir,
        mapcfg: MapConfig = MapConfig(),
        extractor_type: str = "lseg",
        extractor: ExtractorModule = LSegExtractor(),
        accelerate_mapping=True,
        use_sparse_build=True,
    ) -> None:
        self.mapcfg = mapcfg
        self.extractor_type = extractor_type
        self.extractor = extractor
        if not use_sparse_build:
            self.vxlmap = VoxelMapping(
                self.mapcfg, self.extractor, accelerate_mapping=accelerate_mapping
            )
        else:
            self.vxlmap = VoxelMappingSparse(
                self.mapcfg, self.extractor, accelerate_mapping=accelerate_mapping
            )
        self.cam_insc = pinhole.get_camera_intrinsic_matrix(
            mapcfg.screen_h, mapcfg.screen_w, mapcfg.fov
        )
        self.cam_insc_inv = np.linalg.inv(self.cam_insc)

        self.save_dir = save_dir
        self.use_sparse_build = use_sparse_build

    def build(self, obs: Observations):
        rgb = obs.rgb
        depth = obs.depth
        semantic = obs.semantic
        camera_pose = obs.rel_cam_pose
        feats = self.vxlmap.get_feature(rgb)  # torch.Tensor cuda
        self.vxlmap.update(feats, depth, rgb, semantic, camera_pose, self.cam_insc_inv)

    def _return_result(self):
        """return vlmap results for temporaray planning"""
        if self.use_sparse_build:
            return ValueError("Not supported yet")
        featmap = self.vxlmap.featmap
        vxlcnt = np.expand_dims(self.vxlmap.vxl_count, axis=-1)

        indices = np.transpose(np.nonzero(np.any(vxlcnt, axis=-1)))
        feat_values = featmap[tuple(indices.T)]

        return {"indices": indices, "feat_values": feat_values}

    def _save(self):
        if not self.use_sparse_build:
            # This can save 100 times of storage space
            assert len(self.vxlmap.featmap.shape) == 4
            assert len(self.vxlmap.rgbmap.shape) == 4
            assert len(self.vxlmap.gtmap.shape) == 3
            assert len(self.vxlmap.vxl_count.shape) == 3

            featmap = self.vxlmap.featmap
            rgbmap = self.vxlmap.rgbmap
            vxlcnt = np.expand_dims(self.vxlmap.vxl_count, axis=-1)
            gtmap = np.expand_dims(self.vxlmap.gtmap, axis=-1)

            indices = np.transpose(np.nonzero(np.any(vxlcnt, axis=-1)))

            feat_values = featmap[tuple(indices.T)]
            count_values = vxlcnt[tuple(indices.T)].squeeze()
            rgb_values = rgbmap[tuple(indices.T)]
            gt_values = gtmap[tuple(indices.T)].squeeze()
        else:
            indices = self.vxlmap.indices
            feat_values = self.vxlmap.feat_values
            count_values = self.vxlmap.vxl_count
            rgb_values = self.vxlmap.rgb_values
            gt_values = self.vxlmap.gt_values

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            logger.warning(f"{self.save_dir} already exists.")
            input("Press Enter to continue...")
        # Save the indices and values
        save_path = os.path.join(self.save_dir, "sparse_vxl_map.npz")
        np.savez(
            save_path,
            indices=indices,  # [N, 3]
            count_values=count_values,  # [N,]
            feat_values=feat_values,  # [N, feat_dim=512]
            rgb_values=rgb_values,  # [N, 3]
            gt_values=gt_values,  # [N,]
        )
        logger.info(f"{save_path} is saved. (sparse map)")

    @staticmethod
    def _save_npy(save_path, array):
        with open(save_path, "wb") as f:
            np.save(f, array)
            logger.info(f"{save_path} is saved.")
