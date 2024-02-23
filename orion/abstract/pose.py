from typing import List, Optional

import numpy as np
import quaternion
from attr import dataclass, define
from scipy.spatial.transform import Rotation as R

from orion.abstract.interfaces import TextQuery


@define
class Agent2DPose:
    x: int
    z: int
    t: Optional[float] = None  # in radian

    def to_list(self):
        return [self.x, self.z, self.t]

    def from_list(cls, lst):
        if len(lst) == 2:
            return cls(lst[0], lst[1])
        elif len(lst) == 3:
            return cls(lst[0], lst[1], lst[2])
        else:
            raise ValueError("list must be of length 2 or 3")

    @classmethod
    def from_two_pts(cls, src_pos: List[int], tgt_pos: List[int]):
        dx = tgt_pos[0] - src_pos[0]
        dz = tgt_pos[1] - src_pos[1]
        theta = np.arctan2(dz, dx)
        theta_deg = int(np.degrees(theta))
        theta_deg = (theta_deg + 180) % 360 - 180
        return cls(src_pos[0], src_pos[1], np.deg2rad(theta_deg))

    def __getitem__(self, idx):
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.z
        elif idx == 2:
            return self.t
        else:
            raise ValueError("idx must be 0, 1 or 2")

    def __str__(self) -> str:
        if self.t is not None:
            return f"Agent2DPose: x:{self.x},z:{self.z},t:{np.rad2deg(self.t):.3f}"
        else:
            return f"Agent2DPose: x:{self.x},z:{self.z}"


class Agent3DPose:
    rotation: Optional[List[float]]  # [x, y, z, w]
    position: List[float]  # [x, y, z]

    def __init__(
        self,
        position: Optional[List[float]] = None,  # [x, y, z]
        rotation: Optional[List[float]] = None,  # [x, y, z, w]
        pmat: Optional[np.ndarray] = None,
        rmat: Optional[np.ndarray] = None,
    ):
        # either use list or use np.ndarray to initilate
        if position is not None:
            assert len(position) == 3, "position must be a 3-tuple"
            self.position = position
            self.pmat = np.array(self.position, dtype=float)  # [x, y, z]
        elif pmat is not None:
            assert len(pmat) == 3, "position must be a 3-tuple"
            self.pmat = pmat
            self.position = self.pmat.tolist()
        else:
            raise ValueError("position or pos must be given")

        if rotation is not None:
            assert len(rotation) == 4, "rotation must be a 4-tuple"
            self.rotation = rotation
            self.rmat = R.from_quat(self.rotation).as_matrix()  # [x, y, z, w]
        elif rmat is not None:
            assert rmat.shape == (3, 3), "rmat must be a 3x3 matrix"
            self.rmat = rmat
            self.rotation = R.from_matrix(self.rmat).as_quat().tolist()
        else:
            self.rotation = None
            self.rmat = None

    @classmethod
    def from_habitat(cls, habitat_agent_state) -> "Agent3DPose":
        """Create an Agent3DPose from a habitat agent state."""
        position = habitat_agent_state.position.tolist()
        rotation = quaternion.as_float_array(habitat_agent_state.rotation).tolist()[
            ::-1
        ]
        return cls(position=position, rotation=rotation)

    def to_habitat(self):
        """Convert the Agent3DPose to a habitat agent state."""
        return {
            "position": self.position,
            "rotation": self.rotation,
        }

    @classmethod
    def from_numpy(cls, pmat: np.ndarray, rmat: Optional[np.ndarray]) -> "Agent3DPose":
        return cls(pmat=pmat, rmat=rmat)

    @classmethod
    def from_two_pts(cls, src_pos, tgt_pos) -> "Agent3DPose":
        """Create an Agent3DPose from two 3D points."""
        # stay at src pos, face to dst pos
        src_pos = np.array(src_pos)
        tgt_pos = np.array(tgt_pos)
        assert src_pos.shape == (3,)
        assert tgt_pos.shape == (3,)

        dx = tgt_pos[0] - src_pos[0]
        dz = tgt_pos[2] - src_pos[2]
        theta = np.arctan2(-dx, -dz)
        quat = R.from_euler("y", theta).as_quat()
        return cls(position=src_pos.tolist(), rotation=quat.tolist())

    def get_T(self, is_camera_frame: bool = False):
        # get the pose transformation matrix 4x4
        pos = self.pmat
        rmat = self.rmat
        pose = np.eye(4)
        if is_camera_frame:  # camera framec, x right, y down, z forward
            pose[:3, :3] = rmat @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            pose[:3, :3] = rmat  # simulator frame, x right, y up, z backward
        pose[:3, 3] = pos

        return pose

    def to_str(self):
        """Convert the Agent3DPose to a string."""
        return " ".join([str(num) for num in self.position + self.rotation])

    @classmethod
    def from_str(cls, string):
        """Create an Agent3DPose from a string."""
        return cls(
            position=[float(x) for x in string.split(" ")[:3]],
            rotation=[float(x) for x in string.split(" ")[3:]],
        )

    def __str__(self):
        s = "Agent3DPose: "
        x = self.position[0]
        s += "x:{:.3f},".format(x)
        y = self.position[1]
        s += "y:{:.3f},".format(y)
        z = self.position[2]
        s += "z:{:.3f},".format(z)

        if self.rotation is not None:
            qx = self.rotation[0]
            qy = self.rotation[1]
            qz = self.rotation[2]
            qw = self.rotation[3]
            s += "q: [{:.3f},{:.3f},{:.3f},{:.3f}]".format(qx, qy, qz, qw)

        return s


def get_rotation_angle3d(src_pose: Agent3DPose, tgt_pose: Agent3DPose) -> int:
    rot_tgt = tgt_pose.rmat
    rot_src = src_pose.rmat
    rot_src_to_tgt = np.linalg.inv(rot_src) @ rot_tgt
    # Extract the rotation angles from the rotation matrix
    theta_y = np.arctan2(rot_src_to_tgt[0, 2], rot_src_to_tgt[2, 2])
    # Convert the rotation angles from radians to degrees
    theta_y_deg = int(np.degrees(theta_y))
    # wrap to [-180, 180]
    theta_y_deg = (theta_y_deg + 180) % 360 - 180

    return int(theta_y_deg)


def get_rotation_angle2d(
    src_pose: Agent2DPose,
    tgt_pose: Agent2DPose,
) -> int:
    dx = tgt_pose.x - src_pose.x
    dz = tgt_pose.z - src_pose.z
    theta = np.arctan2(dz, dx) - src_pose.t
    theta_deg = int(np.degrees(theta))
    theta_deg = (theta_deg + 180) % 360 - 180

    return int(theta_deg)


def get_l2_dist_2d(src_pose: Agent2DPose, tgt_pose: Agent2DPose) -> float:
    dx = tgt_pose.x - src_pose.x
    dz = tgt_pose.z - src_pose.z
    return np.sqrt(dx**2 + dz**2)


@define
class GoalArea:
    """Goal area in the grid map."""

    x: int
    z: int
    contour: np.ndarray
    height: Optional[float] = None  # with respect to the floor (meter)

    def __getitem__(self, idx):
        if idx == 0:
            return self.x
        elif idx == 1:
            return self.z
        elif idx == 2:
            return self.contour
        elif idx == 3:
            return self.height
        else:
            raise ValueError("idx must be 0, 1, 2 or 3")


@define
class FrontierWayPoint:
    viewpt: Optional[Agent2DPose]
    goalpt: Agent2DPose
    fast_explore: bool = False

    def __str__(self):
        return f"\033[32m FrontierWayPoint\033[m(viewpt: {self.viewpt}, goalpt: {self.goalpt})"

    def __repr__(self) -> str:
        return self.__str__()


@define
class ObjectWayPoint:
    viewpt: Optional[Agent2DPose]
    goalpt: Agent2DPose
    contour: np.ndarray
    height: Optional[float] = None  # with respect to the floor (meter)
    type: Optional[str] = None  # ["mem_pos", "mem_neg", "vlmap", "gpt", "ego_view"]
    text_query: Optional[TextQuery] = None
    score: Optional[float] = None
    object_id: Optional[str] = None

    def __str__(self):
        height = f"{self.height:2f}" if self.height is not None else "None"
        score = f"{self.score:2f}" if self.score is not None else "None"
        return f"\033[32m ObjectWayPoint\033[m(viewpt: {self.viewpt}, goalpt: {self.goalpt}, type: {self.type}, height: {height}, contour: {self.contour.shape}, score: {score})"

    def __repr__(self) -> str:
        return self.__str__()
