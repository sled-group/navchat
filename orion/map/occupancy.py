import numpy as np

from orion.config.my_config import MapConfig
from orion.map.map import Mapping
from orion.utils.geometry import PinholeCameraModel as pinhole


class OccupancyMapping(Mapping):
    # mainly for frontier based exploration
    # also provide explored area
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2
    WALL = 3
    FRONTIER = 4
    UNKNOWN_FREE = 5

    def __init__(self, mapcfg: MapConfig):
        super().__init__(mapcfg=mapcfg)
        self.map = np.zeros((self.num_grid, self.num_grid), dtype=np.uint8)

    def save(self, path):
        np.save(path, self.map)

    def load(self, path):
        self.map = np.load(path)

    def reset_floor(self):
        self.map[self.map == OccupancyMapping.FREE] = OccupancyMapping.UNKNOWN
        self.map[self.map == OccupancyMapping.FRONTIER] = OccupancyMapping.UNKNOWN

    @staticmethod
    def color(value, rgba=False):
        c = None
        if value == OccupancyMapping.UNKNOWN:
            c = [0, 0, 0]
        elif value == OccupancyMapping.FREE:
            c = [255, 255, 255]
        elif value == OccupancyMapping.OCCUPIED:
            c = [255, 0, 0]
        elif value == OccupancyMapping.WALL:
            c = [0, 255, 0]
        elif value == OccupancyMapping.FRONTIER:
            c = [0, 0, 255]
        else:
            raise ValueError("Not supported enum")
        if rgba:
            c.append(255)
        return c

    def update(
        self,
        depth: np.ndarray,
        relative_campose: np.ndarray,
        cam_insc_inv: np.ndarray,
        is_rotate: bool = False,
        camera_height_change: float = 0.0,
    ):
        # depth: [h, w], one image each time

        cam_pts, mask = self.get_point_cloud_from_depth(depth, cam_insc_inv)
        self.cam_pts = cam_pts

        not_ceiling_mask = cam_pts[1, :] > -self.ceiling_height_wrt_camera

        composite_mask = np.logical_and(mask, not_ceiling_mask)

        cam_pts = cam_pts[:, composite_mask]

        # here the wld frame is the first camera frame
        # x right, y down, z forward
        wld_pts = pinhole.cam2wld(cam_pts, cam_pose=relative_campose)
        wld_pts[1, :] -= camera_height_change

        # This is a simple method to get floor mask. It can not handle slope floor.
        # The best way should be using semantic segmentation, like LSeg.
        # However we need a trade-off between speed and accuracy.
        floor_mask = np.logical_and(
            wld_pts[1, :] > self.camera_height - self.agent_height_tolerance,
            wld_pts[1, :] < self.camera_height + self.agent_height_tolerance,
        )

        wall_mask = wld_pts[1, :] < -self.ceiling_height_wrt_camera + 0.2

        grd_xs = np.round(wld_pts[0, :] / self.cell_size + self.num_grid // 2).astype(
            np.int32
        )
        grd_zs = np.round(self.num_grid // 2 - wld_pts[2, :] / self.cell_size).astype(
            np.int32
        )

        # make sure the last occupaied area will not be free anymore.
        # This can avoid more collision
        last_nofree_mask = np.logical_or(
            self.map[grd_zs, grd_xs] == OccupancyMapping.OCCUPIED,
            self.map[grd_zs, grd_xs] == OccupancyMapping.WALL,
        )
        last_wall_mask = self.map[grd_zs, grd_xs] == OccupancyMapping.WALL
        last_map = self.map.copy()

        free_mask = np.logical_and(~last_nofree_mask, floor_mask)

        self.map[grd_zs, grd_xs] = OccupancyMapping.OCCUPIED
        # free_mask = floor_mask

        self.map[grd_zs[free_mask], grd_xs[free_mask]] = OccupancyMapping.FREE
        self.map[grd_zs[wall_mask], grd_xs[wall_mask]] = OccupancyMapping.WALL
        self.map[grd_zs[last_wall_mask], grd_xs[last_wall_mask]] = OccupancyMapping.WALL

        # delete the possible wall area
        floor_mask = np.logical_and(floor_mask, np.logical_not(wall_mask))

        # calculate eight neighbors
        dx = np.array([-1, 0, 1, -1, 1, -1, 0, 1]).reshape(-1, 1)
        dz = np.array([-1, -1, -1, 0, 0, 1, 1, 1]).reshape(-1, 1)

        floor_grds = np.stack([grd_xs[floor_mask], grd_zs[floor_mask]])
        unique_floor_grds = np.unique(floor_grds, axis=1)
        x_grds_floor, z_grds_floor = unique_floor_grds
        neighbor_indices_x = x_grds_floor + dx
        neighbor_indices_z = z_grds_floor + dz

        frontier_mask = (
            np.sum(
                self.map[neighbor_indices_z, neighbor_indices_x]
                == OccupancyMapping.UNKNOWN,
                axis=0,
            )
            > 1
        )
        no_frontier_mask = np.logical_or(
            np.sum(
                self.map[neighbor_indices_z, neighbor_indices_x]
                == OccupancyMapping.WALL,
                axis=0,
            )
            > 1,
            np.sum(
                self.map[neighbor_indices_z, neighbor_indices_x]
                == OccupancyMapping.OCCUPIED,
                axis=0,
            )
            > 3,
        )

        frontier_mask = np.logical_and(frontier_mask, np.logical_not(no_frontier_mask))
        self.map[
            z_grds_floor[frontier_mask], x_grds_floor[frontier_mask]
        ] = OccupancyMapping.FRONTIER

        # # Post-process
        # # This is because the camera height can not be accurate using predicted pose.
        # # So there could be some critical points where large free space turn to occupancy suddenly
        # # We will avoid this
        # increment_occu_mask = np.logical_and(
        #     self.map == OccupancyMapping.OCCUPIED,
        #     np.logical_not(last_map == OccupancyMapping.OCCUPIED),
        # )
        # floor2occu_mask = np.logical_and(
        #     increment_occu_mask, last_map == OccupancyMapping.FREE
        # )
        # if np.sum(floor2occu_mask) > 50 and is_rotate:  # reset
        #     self.map = last_map.copy()
