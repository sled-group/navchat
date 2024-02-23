"""
Interface for habitat ShortestPathFollower and My ShortestPathFollower
"""


import cv2
import numpy as np

from orion import logger
from orion.abstract.pose import Agent2DPose
from orion.navigation.fmm_planner import FMMPlanner
from orion.navigation.waypoint_planner import PointPlanner
from orion.utils import visulization as vis
from orion.utils.geometry import CoordinateTransform


class ShortestPathFollowerBase:
    def is_navigable(self, pose: Agent2DPose):
        raise NotImplementedError

    def get_next_action(self, *args, **kwargs):
        raise NotImplementedError

    def set_traversible_map(self, traversible: np.ndarray):
        raise NotImplementedError

    def get_navigable_mask(self):
        raise NotImplementedError

    def revise_pose(self, pose: Agent2DPose):
        # since the map is constantly changing, we need to revise the pose
        if self.is_navigable(pose):
            return pose
        else:
            logger.info("[PathFollower] Pose is not navigable, revise it auto")
            navi_mask = self.get_navigable_mask()
            navi_mask = vis.get_largest_connected_area(navi_mask)

            new_pose = PointPlanner.plan_reachable_point(
                cen_x=pose.x,
                cen_z=pose.z,
                navigable_mask=navi_mask,
                max_radius=5,
            )
            if new_pose is None:
                logger.info(
                    "[PathFollower] Can not find reachable nearby point, return original pose"
                )
                return pose
            else:
                return Agent2DPose(new_pose.x, new_pose.z, pose.t)


class MyFollower(ShortestPathFollowerBase):
    def __init__(
        self,
        num_rots: int = 360 // 15,
        step_size: int = int(0.25 / 0.05),
        goal_radius: int = int(0.3 / 0.05),
        wheel_radius: int = int(0.2 / 0.05),
    ):
        self.follower = FMMPlanner(num_rots, step_size, goal_radius, wheel_radius)

    def set_traversible_map(self, traversible: np.ndarray):
        self.follower.set_traversible_map(traversible)

    def is_navigable(self, pose: Agent2DPose):
        return self.follower.is_navigable(pose)

    def get_navigable_mask(self):
        return self.follower.original_traversible

    def get_next_action(
        self,
        start: Agent2DPose,
        goal: Agent2DPose,
        pre_collision_dict=None,
        goal_dist=None,
    ):
        _ = self.follower.get_action(
            start, goal, pre_collision_dict=pre_collision_dict, goal_dist=goal_dist
        )
        return _[0]


class HabitatFollower(ShortestPathFollowerBase):
    def __init__(
        self,
        env_sim,
        navigatable_mask: np.ndarray,
        transform_fn: CoordinateTransform,
        goal_radius: float = 0.3,
        return_one_hot: bool = False,
    ):
        self.env_sim = env_sim
        from habitat.tasks.nav.shortest_path_follower import (
            ShortestPathFollower as HabitatShortestPathFollower,
        )

        self.follower = HabitatShortestPathFollower(
            env_sim, goal_radius=goal_radius, return_one_hot=return_one_hot
        )
        self.navigable_mask = navigatable_mask
        self.transform_fn = transform_fn

    def set_traversible_map(self, traversible: np.ndarray):
        pass

    def is_navigable(self, pose: Agent2DPose):
        pos = self.transform_fn.grd2agt_pos(pose)
        return self.env_sim.is_navigable(pos)

    def get_navigable_mask(self):
        return self.navigable_mask

    def get_next_action(self, start: Agent2DPose, goal: Agent2DPose, *args, **kwargs):
        goal_pos = self.transform_fn.grd2agt_pos(goal)
        return self.follower.get_next_action(goal_pos)
