"""
Voxel for VLmap
"""

import numpy as np
import torch

from orion import logger
from orion.abstract.perception import ExtractorModule
from orion.config.my_config import MapConfig
from orion.map.map import Mapping
from orion.utils.geometry import PinholeCameraModel as pinhole


class VoxelMappingSparse(Mapping):
    def __init__(
        self, mapcfg: MapConfig, extractor: ExtractorModule, accelerate_mapping=True
    ):
        super().__init__(mapcfg=mapcfg)
        self.feat_dim = extractor.feat_dim  # feature dim =1 means gt labels
        self.extractor = extractor

        self.feat_values = np.empty(shape=(0, self.feat_dim), dtype=np.float32)
        self.indices = np.empty(shape=(0, 3), dtype=np.int16)
        self.count_values = np.empty(shape=(0), dtype=np.int32)
        self.rgb_values = np.empty(shape=(0, 3), dtype=np.uint8)
        self.gt_values = np.empty(shape=(0), dtype=np.int8)

        self.accelerate_mapping = accelerate_mapping

    def get_feature(self, rgb: np.ndarray):
        return self.extractor.predict(rgb)

    def update(
        self,
        feats: torch.Tensor,
        depth: np.ndarray,
        rgb: np.ndarray,
        semantic: np.ndarray,
        relative_campose: np.ndarray,
        cam_insc_inv: np.ndarray,
    ):
        # depth: [h, w], one image each time
        # feature: [h, w, feat_dim]

        cam_pts, mask = self.get_point_cloud_from_depth(depth, cam_insc_inv)
        not_ceiling_mask = cam_pts[1, :] > -self.ceiling_height_wrt_camera
        composite_mask = np.logical_and(mask, not_ceiling_mask)
        cam_pts = cam_pts[:, composite_mask]

        rgb = rgb.reshape(-1, 3)
        rgb = rgb[composite_mask, :]

        gt_semantic = semantic.reshape(-1)
        gt_semantic = gt_semantic[composite_mask]

        feats = feats.reshape(-1, self.feat_dim)
        composite_mask = torch.from_numpy(composite_mask)
        composite_mask = composite_mask.to(feats.device)  # use gpu to accelerate
        feats = feats[composite_mask, :]
        feats = feats.cpu().numpy()
        composite_mask = None

        # downsample
        cam_pts = cam_pts[:, :: self.downsample_factor]
        rgb = rgb[:: self.downsample_factor, :]
        gt_semantic = gt_semantic[:: self.downsample_factor]
        feats = feats[:: self.downsample_factor, :]

        wld_pts = pinhole.cam2wld(cam_pts, cam_pose=relative_campose)
        vxl_zs = np.round(self.num_grid // 2 - wld_pts[2, :] / self.cell_size).astype(
            np.int32
        )
        vxl_xs = np.round(wld_pts[0, :] / self.cell_size + self.num_grid // 2).astype(
            np.int32
        )
        vxl_ys = np.maximum(
            np.round((self.camera_height - wld_pts[1, :]) / self.cell_size).astype(
                np.int32
            ),
            0,
        )

        if self.accelerate_mapping:
            # get unique voxel indices and the index
            vxl_zxys, vxl_indices = np.unique(
                np.stack([vxl_zs, vxl_xs, vxl_ys], axis=1), axis=0, return_index=True
            )
        else:
            vxl_zxys = np.stack([vxl_zs, vxl_xs, vxl_ys], axis=1)
            vxl_indices = np.arange(vxl_zxys.shape[0])

        for vxl_zxy, vxl_ind in zip(vxl_zxys, vxl_indices):
            self.is_in_grid(vxl_zxy)

            if len(self.indices) == 0:
                self.indices = np.vstack([self.indices, vxl_zxy])
                self.feat_values = np.vstack([self.feat_values, feats[vxl_ind]])
                self.count_values = np.append(self.count_values, 1)
                self.rgb_values = np.vstack([self.rgb_values, rgb[vxl_ind]])
                self.gt_values = np.append(self.gt_values, gt_semantic[vxl_ind])
            else:
                idx = np.where((self.indices == vxl_zxy).all(axis=1))[0]
                if len(idx) > 0:  # already in the indices
                    row_index = idx[0]
                    self.feat_values[row_index] = (
                        self.feat_values[row_index] * self.count_values[row_index]
                        + feats[vxl_ind]
                    ) / (self.count_values[row_index] + 1)
                    self.count_values[row_index] += 1
                    self.rgb_values[row_index] = rgb[vxl_ind]
                    self.gt_values[row_index] = gt_semantic[vxl_ind]
                else:
                    self.indices = np.vstack([self.indices, vxl_zxy])
                    self.feat_values = np.vstack([self.feat_values, feats[vxl_ind]])
                    self.count_values = np.append(self.count_values, 1)
                    self.rgb_values = np.vstack([self.rgb_values, rgb[vxl_ind]])
                    self.gt_values = np.append(self.gt_values, gt_semantic[vxl_ind])

    def is_in_grid(self, vxl_zxy):
        if vxl_zxy[0] < 0 or vxl_zxy[0] >= self.num_grid:
            logger.warning(f"vxl_zxy[0] out of range: {vxl_zxy[0]}")
            vxl_zxy[0] = np.clip(vxl_zxy[0], 0, self.num_grid - 1)
        if vxl_zxy[1] < 0 or vxl_zxy[1] >= self.num_grid:
            logger.warning(f"vxl_zxy[1] out of range: {vxl_zxy[1]}")
            vxl_zxy[1] = np.clip(vxl_zxy[1], 0, self.num_grid - 1)
        if vxl_zxy[2] < 0 or vxl_zxy[2] >= self.num_vxl_height:
            logger.warning(f"vxl_zxy[2] out of range: {vxl_zxy[2]}")
            vxl_zxy[2] = np.clip(vxl_zxy[2], 0, self.num_vxl_height - 1)
