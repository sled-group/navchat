import copy
import itertools
import json
import re
import uuid
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from orion import logger
from orion.abstract.interfaces import TextQuery
from orion.abstract.perception import MaskedBBOX
from orion.abstract.pose import (
    Agent2DPose,
    FrontierWayPoint,
    GoalArea,
    ObjectWayPoint,
    get_rotation_angle2d,
)
from orion.agent_env.chatgpt_control_base import ChatGPTControlBase
from orion.agent_env.habitat.utils import try_action_import_v2
from orion.chatgpt.prompts.agent_functions import *
from orion.config.chatgpt_config import *
from orion.config.my_config import *
from orion.map.occupancy import OccupancyMapping
from orion.navigation.frontier_based_exploration import FrontierBasedExploration
from orion.navigation.waypoint_planner import PointPlanner
from orion.utils import visulization as vis

MOVE_FORWARD, _, TURN_LEFT, TURN_RIGHT, STOP = try_action_import_v2()

REAL_USER = False


class ChatGPTControlORION(ChatGPTControlBase):
    """Do not use the function call version"""

    def execute(self, cmd: dict):
        try:
            if cmd["name"] == "dialog":  # last internal turn
                if "content" not in cmd["args"]:
                    return "Please provide 'content' argument for `dialog` API"
                self.chatgpt.add_assistant_message(cmd["args"]["content"])
            elif cmd["name"] == "move":  # distance>0 forward, distance<0 backward
                if "distance" not in cmd["args"]:
                    return "Please provide 'distance' argument for `move` API"
                if "detect_on" in cmd["args"]:
                    return self.process_move(
                        cmd["args"]["distance"], cmd["args"]["detect_on"]
                    )
                else:
                    return self.process_move(cmd["args"]["distance"])
            elif cmd["name"] == "rotate":  # angle>0 right,  angle<0 left
                if "angle" not in cmd["args"]:
                    return "Please provide 'angle' argument for `rotate` API"
                if "detect_on" in cmd["args"]:
                    return self.process_rotate(
                        cmd["args"]["angle"], cmd["args"]["detect_on"]
                    )
                else:
                    return self.process_rotate(cmd["args"]["angle"])
            elif cmd["name"] == "set_goal":
                if "target" not in cmd["args"]:
                    return "Please provide 'target' argument for `set_goal` API"
                if "prompt" not in cmd["args"]:
                    return "Please provide 'prompt' argument for `set_goal` API"
                if "personlization" not in cmd["args"]:
                    return self.process_set_goal(
                        cmd["args"]["target"], cmd["args"]["prompt"]
                    )
                else:
                    return self.process_set_goal(
                        cmd["args"]["target"],
                        cmd["args"]["prompt"],
                        cmd["args"]["personlization"],
                    )
            elif cmd["name"] == "goto_points":
                # for GPT-4 to plan fine-grained navigation point. points: [(dist, angle)..]
                if "points" not in cmd["args"]:
                    return "Please provide 'points' argument for `goto_points` API"
                if "detect_on" in cmd["args"]:
                    return self.process_goto_points(
                        cmd["args"]["points"], cmd["args"]["detect_on"]
                    )
                else:
                    return self.process_goto_points(cmd["args"]["points"])
            elif cmd["name"] == "change_view":
                if "object_id" not in cmd["args"]:
                    return "Please provide 'object_id' argument for `change_view` API"
                return self.process_change_view(cmd["args"]["object_id"])
            elif cmd["name"] == "goto_object":
                if "object_id" not in cmd["args"]:
                    return "Please provide 'object_id' argument for `goto_object` API"
                return self.process_goto_object(cmd["args"]["object_id"])
            elif cmd["name"] == "update_memory":
                if "object_id" not in cmd["args"]:
                    return "Please provide 'object_id' argument for `update_memory` API"
                if "pos_str_list" not in cmd["args"]:
                    return "Please provide 'pos_str_list' argument for `update_memory` API, if none, it should be empty list"
                if "neg_str_list" not in cmd["args"]:
                    return "Please provide 'neg_str_list' argument for `update_memory` API, if none, it should be empty list"
                return self.process_update_memory(
                    cmd["args"]["object_id"],
                    cmd["args"]["pos_str_list"],
                    cmd["args"]["neg_str_list"],
                )
            elif cmd["name"] == "retrieve_memory":
                # search the episodic memory and vlmap, return <object_id, distance, angle, in_view> tuples
                if "target" not in cmd["args"]:
                    return "Please provide 'target' argument for `retrieve_memory` API"
                if "others" not in cmd["args"]:
                    return "Please provide 'others' argument for `retrieve_memory` API, if none, it should be empty list"
                return self.process_retrieve_memory(
                    cmd["args"]["target"], cmd["args"]["others"]
                )
            elif cmd["name"] == "mask_object":
                if "object_id" not in cmd["args"]:
                    return "Please provide 'object_id' argument for `mask_object` API"
                return self.process_mask_object(cmd["args"]["object_id"])
            elif cmd["name"] == "detect_object":
                if "target" not in cmd["args"]:
                    return "Please provide 'target' argument for `detect_object` API"
                if "prompt" not in cmd["args"]:
                    return "Please provide 'prompt' argument for `detect_object` API"
                return self.process_detect_object(
                    cmd["args"]["target"], cmd["args"]["prompt"]
                )
            elif cmd["name"] == "double_check":
                if "object_id" not in cmd["args"]:
                    return "Please provide 'object_id' argument for `double_check` API"
                return self.process_double_check(cmd["args"]["object_id"])
            elif cmd["name"] == "detect_free_space":
                return self.process_detect_free_space()
            elif cmd["name"] == "detect_object_space":
                return self.process_detect_object_space(cmd["args"]["object_id"])
            elif cmd["name"] == "face_object":
                return self.process_face_object(cmd["args"]["object_id"])
            elif cmd["name"] == "search_object":
                if "target" not in cmd["args"]:
                    return "Please provide 'target' argument for `search_object` API"
                if "prompt" not in cmd["args"]:
                    return "Please provide 'prompt' argument for `search_object` API"
                return self.process_search_object(
                    cmd["args"]["target"], cmd["args"]["prompt"]
                )
            elif cmd["name"] == "update_egoview_info":
                if "object_id_list" not in cmd["args"]:
                    return "Please provide 'object_id_list' argument for `update_egoview_info` API"
                return self.process_update_egoview_info(cmd["args"]["object_id_list"])
            elif cmd["name"] == "retrieve_room":
                return self.process_retrieve_room(cmd["args"]["room_name"])
            elif cmd["name"] == "update_room":
                return self.process_update_room(cmd["args"]["room_name"])
            else:
                return f"Not supported command {cmd['name']}"
        except Exception as e:
            logger.error(e)
            return f"Error: {e}"

    ##### Command Execution #####
    def process_move(self, distance, detect_on=False, refobjwpt=None):
        # return "Sorry `move` API is not available now. Please try use other APIs, such as  `goto_object`, `goto_points` and `search_object` to move."
        if 0 < distance < 10:
            distance = 10
        if -10 < distance < 0:
            distance = -10
        points = [(distance, 0)]
        return_msg = self.process_goto_points(points, detect_on=detect_on)
        if "Can not find" in return_msg:
            return "Can not move properly, you can use `search_object` API to find"
        else:
            return return_msg

    def process_rotate(self, angle, detect_on=False, refobjwpt=None):
        angle = int(angle)
        if abs(angle) == 360:
            angle = 360  # turn around counter-clockwise
        if angle == 0:
            actions = []
        else:
            if angle < 0:
                actions = [TURN_LEFT for _ in range(abs(angle) // self.turn_angle)]
            else:
                actions = [TURN_RIGHT for _ in range(abs(angle) // self.turn_angle)]

        for action in actions:
            self.step(action=action)
            if self.clip_found_goal and detect_on:
                if self.user_text_goal is None:
                    return "Please set goal with `set_goal` API first"
                target = self.user_text_goal.target
                prompt = self.user_text_goal.prompt
                result = self.process_detect_object(
                    target=target,
                    prompt=prompt,
                    skip_false_info=True,
                    refobjwpt=refobjwpt,
                )
                if result is not None and "Found 0 items" not in result:
                    return result

        return "Already rotated {} degrees.".format(angle)

    def process_goto_points(self, points, detect_on=True):
        for p in points:
            if len(p) != 2:
                return "Please provide points as [[dist, angle],...]"
            dist, angle = p
            _pose2d = copy.deepcopy(self.agent_state.pose_2d)
            g = self._egoview2grdview(dist, angle, _pose2d)
            if g is not None:
                logger.info(f"[Command] GOTO Point {g}")
                self.next_pt_to_show = g
                return_msg = self._fast_march_goal(goal=g, detect_on=detect_on)
                if return_msg is not None:
                    return return_msg
            else:
                return "Can not find suitable navigatable place nearby the goal point, try a smalled distance"

        return "Already reached navigatable place nearby the goal point. You can use `detect_object` API to detect if needed."

    def process_goto_object(self, object_id: str):
        if object_id in self.object_id_dict:
            object_waypt = self.object_id_dict[object_id]
            logger.info(f"[Command] GOTO {object_id} {object_waypt}")
            if object_waypt.viewpt is None:
                return (
                    f"Can not automatically find suitable view point for current {object_id}, "
                    f"please try to use low-level movements or ask user for help"
                )
            self.next_pt_to_show: Optional[Agent2DPose] = object_waypt.viewpt
            self.next_contour_to_show: Any = object_waypt.contour
            self.display()
            if (
                object_waypt.type == "vlmap"
            ):  # or self.user_text_goal.target not in object_id:
                detect_on = object_id not in self.reached_vlmap_oids
                print("go to object detect_on:", detect_on)
                if object_id not in self.reached_vlmap_oids:
                    self.reached_vlmap_oids.append(object_id)

                print("fast marching during goto object")
                return_msg = self._fast_march_goal(
                    goal=object_waypt.viewpt, detect_on=detect_on, objwpt=object_waypt
                )
                # can be interrupted by clip filter
                if return_msg is not None and "ego-view detection" in return_msg:
                    if REAL_USER:
                        return (
                            f"When going to {object_id}, ego-view detection interrupts. {return_msg}. "
                            f"You can `double_check` or ask user to confirm. If it's not correct, you can resume go to {object_id} again."
                        )
                    else:
                        return (
                            f"When going to {object_id}, ego-view detection interrupts. {return_msg}. "
                            f"You can `double_check` if it's too far (> 40 units), or ask user to confirm. "
                            f"If it's not correct, you can resume go to {object_id} again."
                        )
                print("face goal during goto object")
                return_msg = self._face_to_goal(
                    goal=object_waypt.goalpt, detect_on=detect_on, objwpt=object_waypt
                )
                if return_msg is not None and "ego-view detection" in return_msg:
                    if REAL_USER:
                        return (
                            f"When going to {object_id}, ego-view detection interrupts. {return_msg}. "
                            f"You can `double_check`  or ask user to confirm. If it's not correct, you can resume go to {object_id} again."
                        )
                    else:
                        return (
                            f"When going to {object_id}, ego-view detection interrupts. {return_msg}. "
                            f"You can `double_check` if it's too far (>40 units), or ask user to confirm. "
                            f"If it's not correct, you can resume go to {object_id} again."
                        )
                self._update_prediction_for_user_simulator(
                    None, self.next_contour_to_show
                )
                return (
                    f"Already reached {object_id}. (Since it is provided by coarse map, "
                    f"may need use `dialog` API to confirm with user or `detect_object` to check)"
                )
            else:
                if (
                    self.user_text_goal.target
                    and self.user_text_goal.target.replace(" ", "_") not in object_id
                ):
                    detect_on = True
                else:
                    detect_on = False
                self._fast_march_goal(goal=object_waypt.viewpt, detect_on=detect_on)
                self._face_to_goal(goal=object_waypt.goalpt, detect_on=detect_on)
                self._update_prediction_for_user_simulator(
                    None, self.next_contour_to_show
                )
                return f"Already reached {object_id}. (Since it is provided by episodic memory, it's likely to be correct, you can confirm with user by `dialog` API if this is the target, detect or look around if this is a related object)"
        else:
            return f"'{object_id}' is not a suitable object id, please try correct object id provided by previous API results"

    def process_change_view(self, object_id: str):
        if "detect" in object_id:
            return f"detect_object_id should use `double_check` API to change the view point, `change_view` is for retrieved object id"
        if object_id in self.object_id_dict:
            object_waypt = self.object_id_dict[object_id]
            logger.info(f"[Command] ChangeView {object_id} {object_waypt}")
            object_waypt = self._egoview_replan(object_waypt)

            if object_waypt.viewpt is None:
                return (
                    f"Can not automatically find suitable view point for current {object_id}, "
                    f"please try to use low-level movements like `goto_points` or ask user for help"
                )
            self.next_pt_to_show = object_waypt.viewpt
            self.next_contour_to_show = object_waypt.contour
            self._fast_march_goal(goal=object_waypt.viewpt)
            self._face_to_goal(goal=object_waypt.goalpt)

            return f"Already go to another viewpoint to look at the {object_id}."
        else:
            return f"'{object_id}' is not a suitable object id, please try correct object id provided by previous API results"

    def process_update_memory(
        self, object_id: str, pos_str_list: List[str], neg_str_list: List[str]
    ):
        if not self.use_memory:
            return "Already updated memory."
        if object_id in self.object_id_dict:
            object_waypt = self.object_id_dict[object_id]
            self.next_contour_to_show = object_waypt.contour
            self.display()
            self.next_contour_to_show = None
            postive_feat = (
                self.map_search.clip_model.encode_text(pos_str_list)
                if len(pos_str_list) > 0
                else None
            )
            negative_feat = (
                self.map_search.clip_model.encode_text(neg_str_list)
                if len(neg_str_list) > 0
                else None
            )
            # contour -> grds
            _grd = np.zeros(
                shape=(self.mapcfg.num_grid, self.mapcfg.num_grid), dtype=np.uint8
            )
            cv2.drawContours(_grd, [object_waypt.contour], -1, 1, -1)
            grds = np.stack(np.where(_grd == 1), axis=1)
            self.object_memory.update(
                neg_feat_val=negative_feat, tgt_feat_val=postive_feat, grd_zxs=grds
            )
            return_msg = f"Already updated {object_id} in memory. "

            if len(pos_str_list) > 0:
                pos_str = ", ".join([f'"{i}"' for i in pos_str_list])
                return_msg += f"It is {pos_str}. "
            if len(neg_str_list) > 0:
                neg_str = ", ".join([f'"{i}"' for i in neg_str_list])
                return_msg += f"It is not {neg_str}. "
                self.process_mask_object(object_id)

            return_msg += " If the current goal is not reached, you may need to `mask_object` to avoid detect again, otherwise you may need `set_goal` for the new goal"
            return return_msg
        else:
            return f"'{object_id}' is not a suitable object id, please try correct object id provided by previous API results."

    def process_retrieve_memory(self, target: str, others: List[str]):
        if target in others:
            others.remove(target)
        all_objects = [target] + others

        all_objects = [i for i in all_objects if i not in ["floor", "wall", "door"]]

        try:
            self.user_text_goal == None
        except:
            return "Please set goal with `set_goal` API first"

        if self.user_text_goal is None:
            return "Please set goal with `set_goal` API first"

        if self.user_text_goal.target != target:
            return (
                f"The set goal target {self.user_text_goal} is not match with the retrieve target {target}, "
                f"please reset goal or use the same target for memory retrieval"
            )

        wall_mask = self.fbe.wall_mask

        nothing_found = True

        return_msg = "\n"
        for item in all_objects:
            # search the episodic memory, get positive and negative map
            if self.user_text_goal.target == item:
                if self.personlization is not None and item in self.personlization:
                    posmem_results, negmem_results = self._memory_retrieve(
                        TextQuery(prompt=self.personlization)
                    )
                else:
                    posmem_results, negmem_results = self._memory_retrieve(
                        TextQuery(prompt=self.user_text_goal.prompt, target=item)
                    )
            else:
                posmem_results, negmem_results = self._memory_retrieve(TextQuery(item))
            posmem_results = self._remove_negative_waypts(
                posmem_results, negmem_results + self.tmp_masked_objwpts
            )
            return_tuple_list = []
            for wpt in posmem_results:
                _, is_in_view = PointPlanner.line_search(
                    self._grdpose.x,
                    self._grdpose.z,
                    wpt.goalpt.x,
                    wpt.goalpt.z,
                    wall_mask,
                    stop_at_wall=True,
                    object_contours=wpt.contour,
                )
                object_id = self._get_object_id(base_str="memory", object_str=item)
                dist, angle = self._grdview2egoview(wpt.goalpt)
                if dist < 35 and -90 < angle < 90 and is_in_view:
                    continue

                is_in_view_str = "in same room" if is_in_view else "in other room"
                return_tuple_list.append(
                    (
                        angle,
                        f"<{object_id}, {dist} units, {angle} degrees, {is_in_view_str}>",
                    )
                )
                wpt.object_id = object_id
                self.object_id_dict[object_id] = wpt

            return_tuple_list_sorted = [
                item[1] for item in sorted(return_tuple_list, key=lambda x: x[0])
            ]
            if len(return_tuple_list_sorted) > 0:
                nothing_found = False
            return_tuple_list_str = json.dumps(return_tuple_list_sorted).replace(
                '"', ""
            )

            if self.user_text_goal.target == item:
                if self.personlization is not None and item in self.personlization:
                    return_msg += f"Found {len(posmem_results)} items for {self.personlization} in good memory: {return_tuple_list_str}\n"
            else:
                return_msg += f"Found {len(posmem_results)} items for {item} in good memory: {return_tuple_list_str}\n"

            # search the vlmap, get possible map
            vlmap_results = self._vlmap_retrieve(TextQuery(item))
            vlmap_results = self._remove_negative_waypts(
                vlmap_results, negmem_results + self.tmp_masked_objwpts
            )
            return_tuple_list = []
            for wpt in vlmap_results:
                _, is_in_view = PointPlanner.line_search(
                    self._grdpose.x,
                    self._grdpose.z,
                    wpt.goalpt.x,
                    wpt.goalpt.z,
                    wall_mask,
                    stop_at_wall=True,
                    object_contours=wpt.contour,
                )
                object_id = self._get_object_id(base_str="possible", object_str=item)
                dist, angle = self._grdview2egoview(wpt.goalpt)
                is_in_view_str = "in same room" if is_in_view else "in other room"
                return_tuple_list.append(
                    (
                        angle,
                        f"<{object_id}, {dist} units, {angle} degrees, {is_in_view_str}>",
                    )
                )
                wpt.object_id = object_id
                self.object_id_dict[object_id] = wpt

            return_tuple_list_sorted = [
                item[1] for item in sorted(return_tuple_list, key=lambda x: x[0])
            ]
            if len(return_tuple_list_sorted) > 0:
                nothing_found = False
            return_tuple_list_str = json.dumps(return_tuple_list_sorted).replace(
                '"', ""
            )
            return_msg += f"Found {len(vlmap_results)} items for {item} in coarse map: {return_tuple_list_str}\n"

        if nothing_found:
            if REAL_USER:
                return_msg += "You can ask user to provide more hints, like nearby object, rough locations, more detailed descriptions, or try to retrieve typical objects related to the target based on commonsense by yourself, or `search_object` to find."
            else:
                return_msg += "You can try to retrieve typical objects related to the target based on commonsense, or `search_object` to find."
        else:
            return_msg += "You can try to `goto_object` to the retrieved object one by one to check"

        return return_msg

    def process_mask_object(self, object_id: str):
        if object_id in self.object_id_dict:
            object_waypt = self.object_id_dict[object_id]
            self.tmp_masked_objwpts.append(object_waypt)
            self.clip_scorer.set_masktime(10)

            for wptid in self.reached_vlmap_oids[-2:]:
                if wptid in self.object_id_dict and wptid != object_id:
                    refobjwpt = self.object_id_dict[wptid]
                    if self._check_2item_close(
                        object_waypt, refobjwpt, 25, 0.5, 0.8
                    ) and self._check_agent_close2contour(refobjwpt.contour):
                        logger.info(
                            f"Mask out {wptid} for the future search at current goal."
                        )
                        self.tmp_masked_objwpts.append(refobjwpt)
            return (
                f"Already mask out {object_id} for the future search at current goal."
            )
        else:
            return f"'{object_id}' is not a suitable object id, please try correct object id provided by previous API results"

    def process_set_goal(
        self, target: str, prompt: str, personlization: Optional[str] = None
    ):
        if target not in prompt:
            return f"prompt must contain target, but '{target}' is not in '{prompt}'. please reset the goal"
        if len(prompt.split()) > 8:
            return f"prompt should be a short decsription of the target, but '{prompt}' is too long. please revise the prompt"

        text_query = TextQuery(prompt=prompt, target=target)
        self.user_text_goal = text_query

        if personlization is not None and target not in personlization:
            return f"personlization should contain target, but '{target}' is not in '{personlization}'. please reset the goal"

        self.personlization = personlization

        self.clip_text_list = ["other", text_query.prompt]
        self.clip_text_embs = self.clip_detector.encode_text(self.clip_text_list)

        self.process_finish_task()

        return f"Already reset goal."

    def process_detect_object(
        self,
        target: Optional[str],
        prompt: str,
        skip_false_info: bool = False,
        refobjwpt: Optional[ObjectWayPoint] = None,
    ):
        if target and target not in prompt:
            return f"prompt must contain target, but '{target}' is not in '{prompt}'. please change the prompt or target"
        if len(prompt.split()) > 8:
            return f"prompt should be a short decsription of the target, but '{prompt}' is too long. please revise the prompt"

        text_query = TextQuery(prompt=prompt, target=target)

        mbboxes: MaskedBBOX = self._detect_with_sam(self.observations.rgb, text_query)
        if bool(mbboxes) is False:
            if not skip_false_info:
                return f"No {text_query.target} can detected by the detection model, please goto the next possible object or `search_object` or ask user for more help"
            else:
                return None

        # TODO: use detected goal area to update the occupancy map, delete false walls

        bbox_to_show = []
        goal_areas = []
        objwpts: List[ObjectWayPoint] = []
        for mbbox in mbboxes:
            goal_area = self.fbe.get_goal_area(mbbox[-1])
            aa = cv2.contourArea(goal_area.contour)
            if target == "nightstand" and (aa > 200 or aa < 15):
                continue
            if goal_area is None:
                continue
            objwpt_tmp = self._egoview_plan([goal_area], text_query)
            if len(objwpt_tmp) == 0:
                continue

            bbox_to_show.append(mbbox)
            goal_areas.append(goal_area)
            objwpts.append(objwpt_tmp[0])
            m = re.search(r"\((.*?)\)", mbbox[1])
            if m:
                objwpts[-1].score = float(m.group(1))

        if len(mbbox) > 0 and len(objwpts) == 0:
            self.clip_scorer.set_masktime(10)  # avoid negative detection

        if len(objwpts) == 1 and refobjwpt is not None:
            # check if the new detection is overlapped as the Vlmap hints, and try to broder the contour area
            if self._check_2item_close(
                objwpts[0], refobjwpt, 25, 0.5, 0.8
            ) and self._check_agent_close2contour(objwpts[0].contour):
                objwpts[0].contour = vis.contour_merge(
                    self.mapcfg.num_grid, objwpts[0].contour, refobjwpt.contour
                )

        return_msg = "\n"
        return_tuple_list = []
        min_dist = 1000
        for wpt in objwpts:
            object_id = self._get_object_id(
                base_str="detect",
                object_str=text_query.target if text_query.target else "object",
            )
            dist, angle = self._grdview2egoview(wpt.goalpt)
            return_tuple_list.append(
                (
                    angle,
                    f"<{object_id}, {dist} units, {angle} degrees, in same room, score {wpt.score}>",
                )
            )
            wpt.object_id = object_id
            if dist < min_dist:
                min_dist = dist
            self.object_id_dict[object_id] = wpt

        return_tuple_list_sorted = [
            item[1] for item in sorted(return_tuple_list, key=lambda x: x[0])
        ]

        return_tuple_list_str = json.dumps(return_tuple_list_sorted).replace('"', "")
        return_msg += f"Found {len(objwpts)} items for {text_query.target} in ego-view detection: {return_tuple_list_str}\n"

        if len(objwpts) == 0:
            if REAL_USER:
                return_msg += "You may need `change_view` to detect again, or goto other possible objects, if the object is masked out you should `search_object` or ask user for help."
            else:
                return_msg += "You may need `change_view` to detect again, or goto other possible objects, or continue to `search_object` or ask user for help."
        elif min_dist > 36:
            return_msg += (
                "The detected object is too far, you may need to `double check`."
            )

        if len(objwpts) > 0:
            self.next_bbox_to_show = MaskedBBOX.from_tuple_list(True, bbox_to_show)
            self.next_contour_to_show = [wpt.contour for wpt in objwpts]
            self._update_prediction_for_user_simulator(
                self.next_bbox_to_show, self.next_contour_to_show
            )
            self.display()
            self.next_contour_to_show = None
            self.next_bbox_to_show = None
        return return_msg

    def process_double_check(self, object_id: str):
        if object_id not in self.object_id_dict:
            return f"'{object_id}' is not a suitable object id, please try correct object id provided by previous API results. or use `detect_object` or `search_object` API to detect again"

        if "detect" not in object_id:
            return f"`double_check` API only support the object id detected by `detect_object` API. Use `change_view` API to change the view point for object id given by `retrieve_memory` API"

        object_waypt = self.object_id_dict[object_id]
        # if not self._check_agent_close2pose2d(object_waypt.viewpt):
        #     # didn't use the viewpt planned before, so go to the viewpt
        #     self.process_goto_object(object_id=object_id)
        # else:

        # re-plan a new viewpt
        object_waypt = self._egoview_replan(object_waypt)
        # self.object_id_dict[object_id] = object_waypt
        # go to the new viewpt
        self.process_goto_object(object_id=object_id)

        print("before: contour", len(object_waypt.contour))

        # redetect
        mbboxes: MaskedBBOX = self._detect_with_sam(
            self.observations.rgb, object_waypt.text_query
        )
        if bool(mbboxes) is False:
            return (
                f"After moving to a new view point, the {object_id} can not be detected. "
                f"This could be the previous detection is false positive or too close to objects to see. You can decide to search other place or double check again"
            )

        # check if the new detection is overlapped as the previous one, and try to broder the contour area
        # We assume the join area is still the object area, intersected area may be too small

        bbox_to_show = []
        detected: bool = False
        for mbbox in mbboxes:
            goal_area = self.fbe.get_goal_area(mbbox[-1])
            if goal_area is None:
                continue
            if self._check_2item_close(goal_area, object_waypt):
                object_waypt.contour = vis.contour_merge(
                    self.mapcfg.num_grid, goal_area.contour, object_waypt.contour
                )
                bbox_to_show.append(mbbox)
                detected = True
                m = re.search(r"\((.*?)\)", mbbox[1])
                if m:
                    object_waypt.score = float(m.group(1))
                break

        print("after: contour", len(object_waypt.contour))

        # check negative detection
        if object_waypt.text_query:
            negative_mem_waypts = self._memory_retrieve(
                object_waypt.text_query, neg_only=True
            )
        else:
            negative_mem_waypts = self._memory_retrieve(
                self.user_text_goal, neg_only=True
            )

        det_waypts = self._remove_negative_waypts(
            [object_waypt], negative_mem_waypts + self.tmp_masked_objwpts
        )
        if len(det_waypts) == 0:
            detected = False
            self.clip_scorer.set_masktime(10)

        if detected:
            self.next_bbox_to_show = MaskedBBOX.from_tuple_list(True, bbox_to_show)
            self.next_contour_to_show = object_waypt.contour
            self._update_prediction_for_user_simulator(
                self.next_bbox_to_show, self.next_contour_to_show
            )
            self.display()
            self.next_bbox_to_show = None
            self.next_contour_to_show = None

            return (
                f"\n After moving to a new view point, the {object_id} is still detected. The detect score becomes {object_waypt.score}."
                f"This could be a correct detection. You can communicate with user using `dialog` or double check again."
            )
        elif len(mbbox) > 0:
            return (
                f"\n After moving to a new view point, the {object_id} can not be detected. "
                f"But new detection is found. You can issue `detect_object` again."
            )
        else:
            return (
                f"\n After moving to a new view point, the {object_id} can not be detected. "
                f"This could be the previous detection is false positive. You can decide to search other place or double check again"
            )

    def process_detect_free_space(self):
        # give the leftmost, rightmost, farest, nearest point of the space
        front_dist, right_dist, back_dist, left_dist = self._free_space_24way_detect()
        return_msg = (
            f"After scan around the room in ego-view, the left wall is around <{left_dist} unit, -90 degrees>, "
            f"the front wall is around <{front_dist} units, 0 degrees>, "
            f"the right wall is around <{right_dist} units, 90 degrees>, "
            f"the back wall is around <{back_dist} units, -180 degrees>"
        )
        return return_msg

    def process_detect_object_space(self, object_id: str):
        # give the leftmost, rightmost, farest, nearest point of the object
        if object_id in self.object_id_dict:
            object_waypt = self.object_id_dict[object_id]
            contour = object_waypt.contour
            self.next_contour_to_show = contour
            self.display()
            self.next_contour_to_show = None
            left_most, right_most, farest, nearest = self._get_contour_shape(contour)
            return_msg = (
                f"After scan the object {object_id} in ego-view, the leftmost point is <{left_most[0]} units, {left_most[1]} degrees>, "
                f"rightmost point is <{right_most[0]} units, {right_most[1]} degrees>, "
                f"farest point is <{farest[0]} units, {farest[1]} degrees>, "
                f"nearest point is <{nearest[0]} units, {nearest[1]} degrees>"
            )
            return return_msg
        else:
            return f"Can not find {object_id}, please try correct object id"

    def process_face_object(self, object_id: str):
        if object_id in self.object_id_dict:
            object_waypt = self.object_id_dict[object_id]
            self.next_contour_to_show = object_waypt.contour
            self._face_to_goal(goal=object_waypt.goalpt)
            dist, angle = self._grdview2egoview(object_waypt.goalpt)
            self.next_contour_to_show = None
            target_str = (
                object_waypt.text_query.target if object_waypt.text_query else "object"
            )
            return (
                f"Already faced to the {object_id} for {target_str}, "
                f"now the ego-view information is <{object_id}, {dist} units, {angle} degrees, in same room>"
            )
        else:
            return f"'{object_id}' is not a suitable object id, please try correct object id provided by previous API results, or use `detect_object` or `search_object` API to detect again"

    def process_update_egoview_info(self, object_id_list: List[str]):
        return_tuple_list = []
        for obj_id in object_id_list:
            if obj_id in self.object_id_dict:
                object_waypt = self.object_id_dict[obj_id]
                _, is_in_view = PointPlanner.line_search(
                    self._grdpose.x,
                    self._grdpose.z,
                    object_waypt.goalpt.x,
                    object_waypt.goalpt.z,
                    self.fbe.wall_mask,
                    stop_at_wall=True,
                    object_contours=object_waypt.contour,
                )
                dist, angle = self._grdview2egoview(object_waypt.goalpt)
                is_in_view_str = "in same room" if is_in_view else "in other room"
                return_tuple_list.append(
                    f"<{obj_id}, {dist} units, {angle} degrees, {is_in_view_str}>"
                )
            else:
                return f"'{obj_id}' is not a suitable object id, please try correct object id"

        return_tuple_list_str = json.dumps(return_tuple_list).replace('"', "")
        return_msg = f"Update current ego-view information: {return_tuple_list_str}"
        return return_msg

    def process_search_object(self, target: str, prompt: str):
        # using frontier-based exploration to search the object around the room

        if not self.use_explore:
            return_msg = self.process_rotate(angle=360, detect_on=True)
            if "Already rotated" in return_msg:
                return (
                    "Frontier-based exploration search process finished. "
                    "If still no target object found, please try to go to neaby objects to detect or ask user for help"
                )
            else:
                return return_msg

        if self.user_text_goal is None:
            return "Please set goal with `set_goal` API first"

        if self.user_text_goal.target != target:
            return (
                f"The set goal target {self.user_text_goal.target} is not matched with the searching target {target}, "
                f"please reset the user goal or use the same target for frontier-based exploration search"
            )

        if target not in prompt:
            return f"prompt must contain target, but '{target}' is not in '{prompt}'. please change the prompt or target"
        if len(prompt.split()) > 8:
            return f"prompt should be a short decsription of the target, but '{prompt}' is too long. please revise the prompt"

        if self.action_generator is None:  # type: ignore
            self.action_generator = self._action_generator()

        if self.fbe.mode == FrontierBasedExploration.InitSpinMode:
            self.fbe.reset(reset_floor=True)
            spins = [TURN_RIGHT for _ in range(360 // self.turn_angle)]
            self.action_generator = itertools.chain(spins, self.action_generator)  # type: ignore
            self.fbe.mode = FrontierBasedExploration.ExploreMode

        for action in self.action_generator:  # type: ignore
            if action == STOP:
                break
            self.step(action=action)
            if self.clip_found_goal:
                result = self.process_detect_object(
                    target=target, prompt=prompt, skip_false_info=True
                )
                if result is not None and "Found 0 items" not in result:
                    return result

        if self.action_generator is not None:
            self.action_generator = None
        return (
            "Frontier-based exploration search process finished. "
            "If still no target object found, please try to go to neaby objects to detect or ask user for help"
        )

    def process_retrieve_room(self, room_name_input: str):
        room_name = room_name_input.lower().replace("'s", " ")
        room_name = " ".join(room_name.split())

        if room_name not in self.room_memory:
            return f"Can not find '{room_name_input}' in the memory, please `retrieve_memory` and go to specific object in the room"
        else:
            pose = self.room_memory[room_name]
            dist, angle = self._grdview2egoview(pose)
            return f"Found '{room_name_input}' in the memory, it has a viewpoint at <{dist} units, {angle} degrees>,  you can use `goto_points` API to go there, then use `detect_object` or `rotate(angle=360, detect_on=True)` to search for the target"

    def process_update_room(self, room_name_input: str):
        room_name = room_name_input.lower().replace("'s", " ")
        room_name = " ".join(room_name.split())

        agent_pose = self.agent_state.pose_2d
        self.room_memory[room_name] = agent_pose
        return (
            f"Already update room information for {room_name_input} with current pose"
        )

    def process_finish_task(self):
        self.clip_scorer.reset()
        self.action_generator = None
        self.next_bbox_to_show = None
        self.next_pt_to_show = None
        self.next_contour_to_show = None

        self.fbe.mode = FrontierBasedExploration.InitSpinMode
        if self.load_existing_occumap:
            self.fbe._reset_to_existing_map()

        print(self.tmp_masked_objwpts)
        print(self.user_text_goal.target.replace(" ", "_"))
        if (
            len(self.tmp_masked_objwpts) > 0
            and self.tmp_masked_objwpts[-1].object_id is not None
            and self.user_text_goal.target.replace(" ", "_")
            in self.tmp_masked_objwpts[-1].object_id
        ):
            self.tmp_masked_objwpts = self.tmp_masked_objwpts[-1:]
        else:
            self.tmp_masked_objwpts = []

        return "reset all states and ready for user to start a new goal"

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
                        yield from self._spin([-150, 180])

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

    def _egoview2grdview(self, dist, angle, pose_2d=None) -> Optional[Agent2DPose]:
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
            cen_x=x + dx,
            cen_z=z + dz,
            navigable_mask=self.navigable_mask,
            max_radius=45,
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
            if (
                self.clip_found_goal
                and detect_on
                and (
                    self._check_goal_in_egoview(goal, objwpt)
                    or all(i > 0.99 for i in self.clip_scorer.prob_list[-6:])
                )
            ):
                target = self.user_text_goal.target
                prompt = self.user_text_goal.prompt
                result = self.process_detect_object(
                    target=target, prompt=prompt, skip_false_info=True, refobjwpt=objwpt
                )
                if result is not None and "Found 0 items" not in result:
                    return result

            best_action = self.follower.get_next_action(
                start=self._grdpose,
                goal=goal,
                pre_collision_dict=self.pre_collision_dict,
            )

            ctt += 1
            # if self.fbe.l2(self._grdpose, goal)<5:
            #     break

            # if self.fbe.l2(self._grdpose, goal)<18 and ctt > 40:
            #     # stuck
            #     break

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
        if REAL_USER:
            height = (
                obj_wpt.height if obj_wpt.height is not None else 0.8
            )  # to make threshold larger
        else:
            height = obj_wpt.height if obj_wpt.height is not None else 0.3
        contour = obj_wpt.contour
        grd_view = self._egoview_plan_viewpt(cen_x, cen_z, height, contour)

        obj_wpt.viewpt = grd_view

        return obj_wpt

    def _egoview_plan_viewpt(self, cen_x, cen_z, height, contour):
        logger.info(
            f"[EgoViewPlan] plan view point for center {cen_x}, {cen_z}, height {height}"
        )
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
        if abs(height - self.agent_height) > 1.0:
            threshold = 32 if not REAL_USER else 20
            if len(contour) <= 10:
                threshold = 38 if not REAL_USER else 25
        elif abs(height - self.agent_height) > 0.7:
            threshold = 25 if not REAL_USER else 20
            if len(contour) <= 10:
                threshold = 30 if not REAL_USER else 25
        elif abs(height - self.agent_height) > 0.4:
            threshold = 20 if not REAL_USER else 15
            if len(contour) <= 10:
                threshold = 25 if not REAL_USER else 20
        else:
            threshold = 14 if not REAL_USER else 10
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
