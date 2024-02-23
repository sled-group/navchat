import numpy as np

from orion.config.my_config import MapConfig


class Mapping:
    def __init__(self, mapcfg: MapConfig):
        self.mapcfg = mapcfg
        self.num_grid = mapcfg.num_grid
        self.cell_size = mapcfg.cell_size
        self.min_depth = mapcfg.min_depth
        self.max_depth = mapcfg.max_depth
        self.ceiling_height_wrt_camera = mapcfg.ceiling_height_wrt_camera
        self.camera_height = mapcfg.camera_height
        self.agent_height_tolerance = mapcfg.agent_height_tolerance
        self.num_vxl_height = mapcfg.num_vxl_height
        self.downsample_factor = mapcfg.downsample_factor

    def update(self, *args, **kwargs):
        return NotImplementedError

    def get_point_cloud_from_depth(self, depth: np.ndarray, cam_insc_inv: np.ndarray):
        """
        Return 3xN array in camera frame, X right, Y down, Z into the screen
        """
        if len(depth.shape) == 3:
            depth = depth.squeeze()

        h, w = depth.shape

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        x = x.reshape((1, -1))[:, :]
        y = y.reshape((1, -1))[:, :]
        z = depth.reshape((1, -1))  # [1, h*w]

        p_2d = np.vstack([x, y, np.ones_like(x)])
        pc = cam_insc_inv @ p_2d
        pc = pc * z  #
        mask = (pc[2, :] > self.min_depth) * (
            pc[2, :] < self.max_depth * 0.99
        )  # avoid non-deteced points
        return pc, mask
