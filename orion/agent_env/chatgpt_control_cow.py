import copy
import itertools
import uuid
from typing import List, Optional, Tuple

import numpy as np

from orion import logger
from orion.abstract.interfaces import TextQuery
from orion.abstract.pose import (
    Agent2DPose,
    GoalArea,
    ObjectWayPoint,
    get_rotation_angle2d,
)
from orion.agent_env.chatgpt_control_base import ChatGPTControlBase
from orion.agent_env.habitat.utils import try_action_import_v2
from orion.chatgpt.api import ChatAPI
from orion.chatgpt.prompts import baseline_cow_prompt
from orion.chatgpt.prompts.agent_functions import *
from orion.config.chatgpt_config import *
from orion.config.my_config import *
from orion.map.occupancy import OccupancyMapping
from orion.navigation.frontier_based_exploration import FrontierBasedExploration
from orion.navigation.waypoint_planner import PointPlanner
from orion.perception.detector.clipgradcam import CLIPGradCAM
from orion.utils import visulization as vis

MOVE_FORWARD, _, TURN_LEFT, TURN_RIGHT, STOP = try_action_import_v2()


class ChatGPTControlCoW(ChatGPTControlBase):
    def __init__(
        self,
        chatgpt_config,
        use_stream=False,
        record_interaction=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            chatgpt_config=chatgpt_config,
            use_stream=use_stream,
            record_interaction=record_interaction,
            *args,
            **kwargs,
        )

        self.chatgpt.message = [
            {"role": "system", "content": baseline_cow_prompt.SYSTEM_PROMPT}
        ]

        self.clip_gradcam = CLIPGradCAM()
        self.target = None

        self.clip_found_goal_count = 0
        self.clip_found_goal_tolerance = 5

    def execute(self, cmd: dict):
        try:
            if cmd["name"] == "dialog":  # last internal turn
                self.chatgpt.add_assistant_message(cmd["args"]["content"])
            elif cmd["name"] == "move":  # distance>0 forward, distance<0 backward
                return self.process_move(cmd["args"]["distance"])
            elif cmd["name"] == "rotate":  # angle>0 right,  angle<0 left
                return self.process_rotate(cmd["args"]["angle"])
            elif cmd["name"] == "search_object":
                if "target" not in cmd["args"]:
                    return "Please use 'target' in the command args"
                return self.process_search_object(cmd["args"]["target"])
            elif cmd["name"] == "goto_points":
                if "points" not in cmd["args"]:
                    return "Please provide 'points' argument for `goto_points` API"
                return self.process_goto_points(cmd["args"]["points"])
            else:
                return f"Not supported command {cmd['name']}"
        except Exception as e:
            logger.error(e)
            return f"Error: {e}"

    ##### Command Execution #####
    def process_move(self, distance, detect_on=False, refobjwpt=None):
        points = [(distance, 0)]
        return_msg = self.process_goto_points(points)
        if "Can not find" in return_msg:
            return "Can not move properly, ask user to provide the postion of the object goal again, or use `search_object`"
        else:
            return return_msg

    def process_rotate(self, angle, detect_on=False, refobjwpt=None):
        angle = int(angle)
        if abs(angle) == 360:
            angle = -360  # turn around counter-clockwise
        if angle == 0:
            actions = []
        else:
            if angle < 0:
                actions = [TURN_LEFT for _ in range(abs(angle) // self.turn_angle)]
            else:
                actions = [TURN_RIGHT for _ in range(abs(angle) // self.turn_angle)]

        for action in actions:
            self.step(action=action)

        return "Already rotated {} degrees.".format(angle)

    def process_search_object(self, target: str):
        # using frontier-based exploration to search the object around the room
        if self.target != target:
            self.clip_scorer.reset()
            self.fbe.mode = FrontierBasedExploration.InitSpinMode
            self.action_generator = None
            self.next_bbox_to_show = None
            self.next_pt_to_show = None
            self.next_contour_to_show = None
            self.target = target
            self.clip_text_list = ["other", target]
            self.clip_text_embs = self.clip_detector.encode_text(self.clip_text_list)

        if self.action_generator is None:
            self.action_generator = self._action_generator()

        if self.fbe.mode == FrontierBasedExploration.InitSpinMode:
            self.fbe.reset(reset_floor=True)
            spins = [TURN_RIGHT for _ in range(360 // self.turn_angle)]
            self.action_generator = itertools.chain(spins, self.action_generator)
            self.fbe.mode = FrontierBasedExploration.ExploreMode

        for action in self.action_generator:
            if action == STOP:
                break
            self.step(action=action)
            if self.clip_found_goal:
                if self.clip_found_goal_count < self.clip_found_goal_tolerance:
                    self.clip_found_goal_count += 1
                    continue
                self.clip_found_goal_count = 0
                pt = self._check_with_clipgrad(target)
                if pt is not None:
                    # add pt into the action generator
                    logger.info(f"clipgradcam, go to {pt}")
                    self.next_pt_to_show = pt
                    self._fast_march_goal(pt)
                    self.clip_scorer.set_masktime(15)

                    return "Found possible object, may ask user whether is correct, if not continue to `search_object`"

        if self.action_generator is not None:
            self.action_generator = None
        return (
            "Frontier-based exploration search process finished. "
            "If still no target object found, please try to ask user for help"
        )

    def process_goto_points(self, points):
        _pose2d = copy.deepcopy(self.agent_state.pose_2d)
        for p in points:
            if len(p) != 2:
                return "Please provide points as [[dist, angle],...]"
            dist, angle = p
            g = self._egoview2grdview(dist, angle, _pose2d)
            if g is not None:
                logger.info(f"[Command] GOTO Point {g}")
                self.next_pt_to_show = g
                self._fast_march_goal(goal=g)
                break
            else:
                return "Can not find suitable navigatable place nearby the goal point, try a smalled distance"

        return "Already reached navigatable place nearby the goal point. You can use `search_object` or ask user for help"

    def _check_with_clipgrad(self, target: str):
        rgb = self.observations.rgb
        depth = self.observations.depth
        pt = self.clip_gradcam.predict(rgb, target)
        if pt is None:
            return None
        else:
            dist = depth[pt[1], pt[0]][0] / 0.05
            delta = pt[0] - rgb.shape[1] // 2
            angle = np.arctan2(delta, rgb.shape[1] // 2)
            logger.info("clipgradcam: dist {}, angle {}".format(dist, angle))
            grdpt = self._egoview2grdview(dist, angle)
            return grdpt

    def step(self, action):
        super().step(action)
        if self.clip_text_list:
            self._detect_with_clip(self.observations.rgb)

    #### Utils Funciton ####
    @property
    def clip_found_goal(self):
        return self.clip_scorer.found_goal

    def _action_generator(self):
        fbe_waypt = self._plan_fwpt()
        while fbe_waypt is not None and self.use_explore:
            # fast explore: go to goalpt directly
            if fbe_waypt.fast_explore:
                if fbe_waypt.goalpt is not None:
                    count = self.fbe.fbecfg.fast_explore_forwardcount
                    best_action = self._get_next_action(
                        fbe_waypt.goalpt, new_goal_dist=10
                    )
                    while best_action != STOP and count > 0:
                        yield best_action
                        if best_action == MOVE_FORWARD:
                            count -= 1
                        best_action = self._get_next_action(
                            fbe_waypt.goalpt, new_goal_dist=10
                        )

                    if (
                        self.fbe.l2(fbe_waypt.goalpt, self._grdpose)
                        < self.fbe.fbecfg.dist_large_thres
                    ):
                        yield from self._spin([-90, 180])

            # slow explore: go to viewpt, look around , then go to goalpt, look around
            # usually used for collect frames for vlmap
            else:
                if fbe_waypt.viewpt is not None:
                    best_action = self._get_next_action(fbe_waypt.viewpt)
                    while best_action != STOP:
                        yield best_action
                        best_action = self._get_next_action(fbe_waypt.viewpt)
                    yield from self._spin([-90, 180])

                if fbe_waypt.goalpt is not None:
                    best_action = self._get_next_action(
                        fbe_waypt.goalpt, new_goal_dist=10
                    )
                    while best_action != STOP:
                        yield best_action
                    best_action = self._get_next_action(
                        fbe_waypt.goalpt, new_goal_dist=10
                    )
                    yield from self._spin([-90, 180])

            self.next_pt_to_show = None
            self.next_contour_to_show = None
            fbe_waypt = self._plan_fwpt()

    def _plan_fwpt(self):
        self.follower.set_traversible_map(self.fbe.traversable_map)
        fbe_waypt = self.fbe.plan(
            self.navigable_mask, with_viewpoint=not self.fast_explore
        )
        return fbe_waypt

    def _spin(self, angles: List[int]):
        if isinstance(angles, int):
            angles = [angles]
        for angle in angles:
            rot_num = angle // self.turn_angle
            while rot_num != 0:
                if rot_num > 0:
                    yield TURN_RIGHT
                    rot_num -= 1
                else:
                    yield TURN_LEFT
                    rot_num += 1

    def _egoview2grdview(self, dist, angle, pose_2d=None) -> Agent2DPose:
        """convert egocentric view to global view"""
        dist = int(dist)
        angle = int(angle)
        if pose_2d is None:
            x, z, t = self.agent_state.pose_2d.to_list()
        else:
            x, z, t = pose_2d.to_list()
        t = t + np.deg2rad(angle)
        dx = dist * np.cos(t)
        dz = dist * np.sin(t)
        return PointPlanner.plan_reachable_point(
            cen_x=x + dx, cen_z=z + dz, navigable_mask=self.navigable_mask
        )

    def _grdview2egoview(self, grd_view: Agent2DPose) -> Tuple[int, int]:
        """convert global view to egocentric view"""
        x, z, t = self.agent_state.pose_2d.to_list()
        dx = grd_view.x - x
        dz = grd_view.z - z
        dist = int(np.sqrt(dx**2 + dz**2))
        angle = get_rotation_angle2d(
            src_pose=self.agent_state.pose_2d, tgt_pose=grd_view
        )
        return dist, angle

    def _free_space_24way_detect(self):
        # shoot rays from the agent to the wall in 24 directions, get the distance to the obstacles
        x, z, t = self.agent_state.pose_2d.to_list()
        navigation_map = self.navigable_mask
        ways = 24
        delta = 360 // 24
        dist = []
        for idx in range(ways):
            angle = idx * delta - delta * 3
            d = 1
            dx = d * np.cos(np.deg2rad(angle) + t)
            dz = d * np.sin(np.deg2rad(angle) + t)
            while navigation_map[int(z + dz), int(x + dx)] == 1:
                d += 1
                dx = d * np.cos(np.deg2rad(angle) + t)
                dz = d * np.sin(np.deg2rad(angle) + t)
            dist.append(d)

        front_dist = int(np.mean(dist[0:6]))
        right_dist = int(np.mean(dist[6:12]))
        back_dist = int(np.mean(dist[12:18]))
        left_dist = int(np.mean(dist[18:24]))

        return front_dist, right_dist, back_dist, left_dist

    def _fast_march_goal(self, goal: Agent2DPose, detect_on=False, objwpt=None):
        """
        goal: the next point the agent need to go to
        detect_on: whether to detect object when the agent is moving
        objwpt: the object way point, used to check whether the goal is in the ego-view
        """
        if goal is None:
            return
        logger.info(f"[SubCommand] FastMarchGoal {goal}")
        best_action = self.follower.get_next_action(
            start=self._grdpose, goal=goal, pre_collision_dict=self.pre_collision_dict
        )
        ctt = 0
        while best_action != STOP:
            self.step(action=best_action)

            best_action = self.follower.get_next_action(
                start=self._grdpose,
                goal=goal,
                pre_collision_dict=self.pre_collision_dict,
            )

            ctt += 1

            if self.fbe.l2(self._grdpose, goal) < 20 and ctt > 40:
                # stuck
                break

    def _check_goal_in_egoview(
        self, goal: Agent2DPose, objwpt: Optional[ObjectWayPoint] = None
    ):
        if objwpt is None:
            true_goal = goal
            contour = None
        else:
            true_goal = objwpt.goalpt
            contour = objwpt.contour

        dist, angle = self._grdview2egoview(true_goal)
        if dist < 32 and abs(angle) < self.FOV / 2:
            _, is_in_view = PointPlanner.line_search(
                self._grdpose.x,
                self._grdpose.z,
                true_goal.x,
                true_goal.z,
                self.fbe.wall_mask,
                stop_at_wall=True,
                object_contours=contour,
            )
            if is_in_view:
                return True

        return False

    def _face_to_goal(self, goal: Agent2DPose, detect_on=False, objwpt=None):
        logger.info(f"[SubCommand] FaceToGoal {goal}")
        angle = get_rotation_angle2d(src_pose=self.agent_state.pose_2d, tgt_pose=goal)
        result = self.process_rotate(angle, detect_on=detect_on, refobjwpt=objwpt)
        return result

    def _get_object_id(self, base_str: str, object_str: str):
        base_str = base_str.strip().replace(" ", "_")
        object_str = object_str.strip().replace(" ", "_")
        standard_uuid = uuid.uuid4()
        uid = str(standard_uuid.hex)[:5]
        while f"{base_str}_{object_str}_{uid}" in self.object_id_dict:
            standard_uuid = uuid.uuid4()
            uid = str(standard_uuid.hex)[:5]
        return f"{base_str}_{object_str}_{uid}"

    def _egoview_plan(
        self, goal_areas: List[GoalArea], text_query: TextQuery
    ) -> List[ObjectWayPoint]:
        det_waypts: List[ObjectWayPoint] = []
        for goal_area in goal_areas:
            cen_x = goal_area.x
            cen_z = goal_area.z
            height = goal_area.height
            contour = goal_area.contour

            grd_view = self._egoview_plan_viewpt(cen_x, cen_z, height, contour)
            grd_goal = Agent2DPose(x=cen_x, z=cen_z, t=0)
            det_waypts.append(
                ObjectWayPoint(
                    viewpt=grd_view,
                    goalpt=grd_goal,
                    contour=contour,
                    height=height,
                )
            )

        for pt in det_waypts:
            pt.type = "ego_view"
            pt.text_query = text_query

        negative_mem_waypts = self._memory_retrieve(text_query, neg_only=True)

        det_waypts = self._remove_negative_waypts(
            det_waypts, negative_mem_waypts + self.tmp_masked_objwpts
        )

        return det_waypts

    def _egoview_replan(self, obj_wpt: ObjectWayPoint):
        cen_x = obj_wpt.goalpt.x
        cen_z = obj_wpt.goalpt.z
        height = (
            obj_wpt.height if obj_wpt.height is not None else 0.3
        )  # to make threshold larger
        contour = obj_wpt.contour
        grd_view = self._egoview_plan_viewpt(cen_x, cen_z, height, contour)

        obj_wpt.viewpt = grd_view

        return obj_wpt

    def _egoview_plan_viewpt(self, cen_x, cen_z, height, contour):
        if self.use_gt_pose:
            navigable_mask = np.logical_and(
                self.navigable_mask, self.fbe.occu_map() == OccupancyMapping.FREE
            )
        else:
            navigable_mask = self.navigable_mask
        navigable_mask = vis.get_largest_connected_area(navigable_mask, erosion_iter=2)
        wall_mask = self.fbe.wall_mask
        # remove false wall mask using the contour,
        # since the contour can be seen by agent, there should be no wall in between
        if contour is not None:
            for pt in contour:
                pt = pt.squeeze()
                line, _ = PointPlanner.line_search(
                    self._grdpose.x,
                    self._grdpose.z,
                    pt[0],
                    pt[1],
                )
                for p in line:
                    wall_mask[int(p[1]), int(p[0])] = 0

        # if the object is too high or too low, then the view point shoule be a little far away
        if abs(height - self.agent_height) > 0.7:
            threshold = 25
            if len(contour) <= 10:
                threshold = 30
        elif abs(height - self.agent_height) > 0.4:
            threshold = 20
            if len(contour) <= 10:
                threshold = 25
        else:
            threshold = 14
        logger.info(
            f"[EgoViewPlan] set threshold: {threshold} for the egodetected object"
        )
        grd_view = PointPlanner.plan_view_point_with_contour(
            cen_x,
            cen_z,
            navigable_mask,
            wall_mask,
            contour,
            threshold=threshold,
        )
        return grd_view

    def _get_contour_shape(self, contour: np.ndarray):
        rels = []
        for pt in contour:
            pt = pt.squeeze()
            rel_ang = get_rotation_angle2d(
                src_pose=self.agent_state.pose_2d,
                tgt_pose=Agent2DPose(x=pt[0], z=pt[1]),
            )
            rel_len = self.fbe.l2(
                self.agent_state.pose_2d, Agent2DPose(x=pt[0], z=pt[1])
            )
            rels.append((int(rel_len), rel_ang))

        left_most = sorted(rels, key=lambda x: x[1])[0]
        right_most = sorted(rels, key=lambda x: x[1])[-1]
        farest = sorted(rels, key=lambda x: x[0])[-1]
        nearest = sorted(rels, key=lambda x: x[0])[0]

        return left_most, right_most, farest, nearest
