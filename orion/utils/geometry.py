"""
[sim/agt] 3D simulator coordinate system (XYZ): x: right, y: up, z: backward. origin is where the agent spawns in a scene.
Maybe given by simulator like habitat_sim.
Rotate Left is positive, Rotate Right is negative.

[cam] 3D agent camera coordinate system (XYZ): x: right, y: down, z: forward. origin is where the agent spawns in a scene.
[pxl] 2D camera coordinate system (XY): x: right, y: down. origin is the top left corner of the image.
Rotate Left is negative, Rotate Right is positive.

[grd] 2D map coordinate system (XZ): x: right, z: down. origin is the top left corner of the grid top-down map.
[vxl] 3D map coordinate system (XYZ): x: right, y: up, z: backward. origin is the top left corner of the voxel map.

"""

from typing import List

import numpy as np

from orion.abstract.pose import Agent2DPose, Agent3DPose


class CoordinateTransform:
    def __init__(
        self, num_grd: int, cell_size: float, init_agtpose: Agent3DPose  # meter
    ):
        """
        Assume initial agent position is centered at (0, 0, 0) in voxel map, facing the negative z axis (to the front).
        initial simulator agent state is given by simlulator like habitat_sim. But non-applicable in real world.
        """
        self.cell_size = cell_size
        self.num_grid = num_grd

        self.init_agtpose = init_agtpose

        self.init_agtT = self.init_agtpose.get_T(
            is_camera_frame=False
        )  # z back, x right, y up
        self.init_agtTinv = np.linalg.inv(self.init_agtT)

        self.init_camT = self.init_agtpose.get_T(
            is_camera_frame=True
        )  # z forward, x right, y down
        self.init_camTinv = np.linalg.inv(self.init_camT)

    def get_relative_agtpose(self, new_agtpose: Agent3DPose) -> np.ndarray:
        new_agtT = new_agtpose.get_T(is_camera_frame=False)
        rel_agtT = self.init_agtTinv @ new_agtT
        return rel_agtT

    def get_relative_campose(self, new_agtpose: Agent3DPose) -> np.ndarray:
        # return T matrix
        new_camT = new_agtpose.get_T(is_camera_frame=True)
        cam_exsc_inv = self.init_camTinv @ new_camT
        return cam_exsc_inv

    def grd2agt_pos(self, pose2d: Agent2DPose) -> np.ndarray:
        grd_x, grd_z = pose2d.x, pose2d.z
        return self.vxl2agt_pos(grd_x, 0, grd_z)

    def vxl2agt_pos(self, vxl_x: int, vxl_y: int, vxl_z: int) -> np.ndarray:
        # we only need pos here in repo, no need to consider rotation yet
        agt_x = (vxl_x - self.num_grid // 2) * self.cell_size
        agt_z = (vxl_z - self.num_grid // 2) * self.cell_size
        agt_y = vxl_y * self.cell_size
        agt_pos = self.init_agtT @ np.array([agt_x, agt_y, agt_z, 1]).reshape(-1, 1)
        agt_pos = agt_pos[:3, 0]
        return agt_pos  # just position, too simple, no need to wrap it into Agent3DPose

    def agt2grd_pose(self, agtpose: Agent3DPose) -> Agent2DPose:
        cur_pos = self.init_agtTinv @ np.append(agtpose.pmat, 1).reshape(-1, 1)
        agt_z = cur_pos[2, 0]
        agt_x = cur_pos[0, 0]

        grd_x = self.num_grid // 2 + int(agt_x / self.cell_size)
        grd_z = self.num_grid // 2 + int(agt_z / self.cell_size)

        rmat = agtpose.rmat
        if rmat is None:
            return Agent2DPose(x=grd_x, z=grd_z)

        # get the rotation in the grid map
        rmat_ref = np.linalg.inv(self.init_agtpose.rmat) @ agtpose.rmat
        rtheta_ref = np.arctan2(rmat_ref[0, 2], rmat_ref[0, 0])
        t = -rtheta_ref - np.pi / 2
        t = (t + np.pi) % (2 * np.pi) - np.pi
        return Agent2DPose(x=grd_x, z=grd_z, t=t)


class PinholeCameraModel:
    # cam_extrinsic = inv(cam_pose）
    # pxl_coord = cam_intrinsic @ cam_extrinsic @ wld_coord
    # pxl_coord = cam_intrinsic @ cam_coord / depth
    # cam_coord = cam_extrinsic @ wld_coord
    # general transformation for pinhole camera model

    @staticmethod
    def get_camera_intrinsic_matrix(h, w, fov=90, desire_h=None, desire_w=None):
        cam_mat = np.eye(3)
        cam_mat[0, 0] = cam_mat[1, 1] = w / (2.0 * np.tan(np.deg2rad(fov / 2)))
        cam_mat[0, 2] = w / 2.0
        cam_mat[1, 2] = h / 2.0

        if desire_h is not None and desire_w is not None:
            # Calculate the scale factors in each dimension
            scale_x = desire_w / w
            scale_y = desire_h / h

            # Create the scaling matrix
            scale_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])

            # Calculate the new camera intrinsic matrix
            cam_mat = np.matmul(scale_matrix, cam_mat)

        return cam_mat

    @staticmethod
    def cam2pxl(cam_pts, cam_insc):
        # cam_pts: (3, N)
        if cam_pts.shape == (3,):
            single_point = True
            cam_pts = cam_pts.reshape((3, 1))
        else:
            single_point = False

        new_p = np.matmul(cam_insc, cam_pts)
        new_p = new_p / new_p[2, 0]
        uv = np.floor(new_p[:2, :] + 0.5).astype(np.int32)

        if single_point:
            return uv[0, 0], uv[1, 0]
        else:
            return uv  # 2xN

    @staticmethod
    def pxl2cam(pxl_pts, cam_insc, depth):
        # pxl_pts: [2xN]
        # depth: [N]
        if pxl_pts.shape == (2,):
            single_point = True
            pxl_pts = pxl_pts.reshape((2, 1))
        else:
            single_point = False

        homoge_pxl_pts = np.vstack([pxl_pts, np.ones((1, pxl_pts.shape[1]))])
        cam_pts = np.matmul(np.linalg.inv(cam_insc), homoge_pxl_pts) * depth

        if single_point:
            return cam_pts[:, 0]  # (3,)
        else:
            return cam_pts  # 3xN

    @staticmethod
    def cam2wld(cam_pts, cam_exsc=None, cam_pose=None):
        """
        cam_pts: [3, N] tensor
        inv(cam_extrinsic) = camera pose, i.e., camera [R, T] 4x4 homomatrix to world coordinate system)
        either cam_exsc or cam_pose should be given
        """
        if cam_pts.shape == (3,):
            single_point = True
            cam_pts = cam_pts.reshape((3, 1))
        else:
            single_point = False

        cam_pts_homo = np.vstack([cam_pts, np.ones((1, cam_pts.shape[1]))])

        if cam_pose is not None:
            cam_exsc_inv = cam_pose
        elif cam_exsc is not None:
            cam_exsc_inv = np.linalg.inv(cam_exsc)
        else:
            raise ValueError("Either cam_exsc or cam_pose should be given")

        wld_pts_homo = np.matmul(cam_exsc_inv, cam_pts_homo)

        if single_point:
            return wld_pts_homo[:3, 0]  # (3, )
        else:
            return wld_pts_homo[:3, :]  # [3, N]

    @staticmethod
    def wld2cam(wld_pts, cam_exsc=None, cam_pose=None):
        if wld_pts.shape == (3,):
            single_point = True
            wld_pts = wld_pts.reshape((3, 1))
        else:
            single_point = False

        wld_pts_homo = np.vstack([wld_pts, np.ones((1, wld_pts.shape[1]))])

        if cam_exsc is not None:
            cam_exsc = cam_exsc
        elif cam_pose is not None:
            cam_exsc = np.linalg.inv(cam_pose)

        cam_pts_homo = np.matmul(cam_exsc, wld_pts_homo)

        if single_point:
            return cam_pts_homo[:3, 0]
        else:
            return cam_pts_homo[:3, :]

    @staticmethod
    def wld2pxl(wld_pts, cam_exsc, cam_insc, cam_pose):
        cam_pts = PinholeCameraModel.wld2cam(wld_pts, cam_exsc, cam_pose)
        pxl_pts = PinholeCameraModel.cam2pxl(cam_pts, cam_insc)
        return pxl_pts

    @staticmethod
    def pxl2wld(pxl_pts, depth, cam_exsc, cam_insc, cam_pose):
        cam_pts = PinholeCameraModel.pxl2cam(pxl_pts, cam_insc, depth)
        wld_plt = PinholeCameraModel.cam2wld(cam_pts, cam_exsc, cam_pose)
        return wld_plt


def undistort(
    points: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: List[float],
    iter_num=1,
):
    # points (2, )
    k1, k2, p1, p2, k3 = distortion
    x, y = points
    x0 = (x - cx) / fx
    y0 = (y - cy) / fy
    for _ in range(iter_num):
        r2 = x**2 + y**2
        k_inv = 1 / (1 + k1 * r2**2 + k2 * r2**4 + k3 * r2**6)
        delta_x = 2 * p1 * x * y + p2 * (r2**2 + 2 * x**2)
        delta_y = p1 * (r2**2 + 2 * y**2) + 2 * p2 * x * y
        x = (x0 - delta_x) * k_inv
        y = (y0 - delta_y) * k_inv
    return np.array((x * fx + cx, y * fy + cy))
