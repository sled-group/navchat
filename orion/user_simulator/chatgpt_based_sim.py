import json
import random
import re
from typing import List, Optional, Tuple
import numpy as np
from copy import deepcopy

from orion.abstract.pose import Agent2DPose
from orion.config.my_config import *
from orion.utils import visulization as vis

from orion import logger
from orion.user_simulator.user_goal import UserGoal
from orion.config.chatgpt_config import *
from orion.user_simulator.base import UserSimulatorBase

import random

random.seed(1)


class CountourMaskPrediction:
    NONE = 0
    CONTOUR_ONLY = 1
    BOTH = 2
    WHOLE_IMAGE = 3

    """Detected/Retrieved Results for User Simulator to judge goal reached """

    def __init__(
        self,
        predict_contours: Optional[List[np.ndarray]] = None,
        predict_masks: Optional[List[np.ndarray]] = None,
    ):
        self.predict_contours = (
            deepcopy(predict_contours) if predict_contours is not None else []
        )
        logger.info(
            f"[User Simulator] predict_contours: {[c.shape for c in self.predict_contours]}"
        )
        self.predict_masks = (
            deepcopy(predict_masks) if predict_masks is not None else []
        )
        logger.info(
            f"[User Simulator] predict_masks: {[m.shape for m in self.predict_masks]}"
        )

        if len(self.predict_contours) == 0 and len(self.predict_masks) == 0:
            self.flag = CountourMaskPrediction.NONE
        elif len(self.predict_contours) > 0 and len(self.predict_masks) == 0:
            self.flag = CountourMaskPrediction.CONTOUR_ONLY
        elif len(self.predict_contours) == 0 and len(self.predict_masks) > 0:
            if np.sum(predict_masks[0]) == np.prod(predict_masks[0].shape):
                self.flag = CountourMaskPrediction.WHOLE_IMAGE
                self.predict_contours = [None for _ in range(len(predict_masks))]
            else:
                logger.warning(
                    f"[User Simulator] predict_contours is empty but predict_masks is not empty!!!"
                )
                self.flag = CountourMaskPrediction.NONE
        else:
            self.flag = CountourMaskPrediction.BOTH


class ChatGPTUserSimulator(UserSimulatorBase):
    def step(
        self,
        agent_response: str,
        agtpose: Agent2DPose,
        semantic_img: np.ndarray,
        agt_predict: CountourMaskPrediction,
        step_count: int,
        first_turn: bool = False,
    ) -> Tuple[bool, bool, bool, bool, str]:
        self.agtpose = agtpose
        semantic_img = self._cvt_semid2clsid(semantic_img)
        current_goal = deepcopy(self.goal_gen.current_goal)

        logger.info(f"[User Simulator] current_goal: {current_goal}")
        ins = self.topo_graph.instance_dict[current_goal.goal]
        logger.info(f"[User Simulator] current goal instance: {ins}")
        logger.info(f"[User Simulator] current agtpose: {agtpose}")

        success_by_semimg = self._eval_with_semantic_img(
            current_goal, agtpose, semantic_img
        )
        logger.info(
            f"[User Simulator] Eval with semantic image, goal reached: {success_by_semimg}"
        )

        if agt_predict.flag != CountourMaskPrediction.NONE:
            success_by_predict, robot_detection_list = self._eval_with_agt_predict(
                current_goal, agtpose, semantic_img, agt_predict
            )
        else:
            success_by_predict, robot_detection_list = (
                False,
                [],
            )  # no detection input, like COW
        logger.info(
            f"[User Simulator] Eval with prediction, goal reached: {success_by_predict}"
        )

        if first_turn:
            goal_reached = False
        else:
            goal_reached = success_by_semimg or success_by_predict

        maxtry_reached, task_finished = self.goal_gen.step(
            goal_reached,
            steps=step_count - self.last_step_count,
            first_turn=first_turn,
            agtpose=agtpose,
        )
        self.last_step_count = step_count

        if task_finished:
            return True, False, False, False, "That's all for today. Thank you for your help."

        egoview_info_goal = self._get_egoview_info_for_goal(
            self.goal_gen.current_goal, agtpose
        )  # current goal may change if goal_reached is True

        if goal_reached or maxtry_reached:
            is_new_goal = True
        else:
            is_new_goal = False

        is_new_round = self.goal_gen._is_new_round
        if is_new_round:
            self.goal_gen._is_new_round = False


        user_msg = self._wrap_user_message_for_chatgpt(
            agent_response=agent_response,
            goal_reached=goal_reached,
            maxtry_reached=maxtry_reached,
            active_goal=self.goal_gen.current_goal,
            egoview_info_goal=egoview_info_goal,  # # hints for position
            robot_detection_list=robot_detection_list,  # for language correction, a list of object_id or object name
            first_turn=first_turn,
        )


        logger.info(f"[User Simulator] input: \n{user_msg}")
        self.chatgpt.add_user_message(user_msg)
        response = self.chatgpt.get_system_response()
        self.chatgpt.add_assistant_message(response)
        response = self._post_process(response)
        print("chat_history:", len(self.chatgpt.messages))

        return task_finished, is_new_goal, is_new_round, goal_reached, response


    def _wrap_user_message_for_chatgpt(
        self,
        agent_response: str,
        goal_reached: bool,
        maxtry_reached: bool,
        active_goal: UserGoal,
        egoview_info_goal: dict,
        robot_detection_list: dict,
        first_turn: bool = False,
    ) -> str:
        # prepare all possible info for chatgpt too choose
        return_msg = f"Robot Utterance: {agent_response}\n"
        return_msg += f"Function Message: \n"

        ins = self.topo_graph.instance_dict[active_goal.goal]
        _goal_info = {
            "object_id": ins.id,
            "object_name": ins.name,
            # "room_id": ins.room_id,
            "room_name": ins.room_desc,
            "ego_view_info": egoview_info_goal["tuple_str"]
            if self.category in ["instruction", "mixed"]
            else "",
            "rough_route": " | ".join(egoview_info_goal["route"])
            if self.category in ["instruction", "mixed"] and len(egoview_info_goal["route"])<=3
            else "",
            "nearby_objects": " | ".join(random.sample(ins.nearby_obj, min(len(ins.nearby_obj), 2)))
            if ins.nearby_obj and self.category in ["landmark", "mixed"]
            else "",
            "description": " | ".join(random.sample(ins.object_desc, 1))
            if ins.object_desc and self.category in ["description", "mixed"]
            else "",
            "explaination": ins.explain
            if ins.explain and self.category in ["description", "mixed"]
            else "",
            "special_attr": ins.attr
            if ins.attr and self.category in ["correction", "mixed"]
            else "",
            "num_trial": active_goal.trial if not first_turn else 0,
            "num_round": active_goal.round,
        }
        goal_info = {}
        for k, v in _goal_info.items():
            if v!="":
                goal_info[k] = v
            elif self.category in ["landmark"] and k =="nearby_objects":
                goal_info[k] = v
            else:
                pass


        output_dic = {}
        if goal_reached or maxtry_reached or first_turn:
            output_dic["current_goal"] = {}
        else:
            output_dic["current_goal"] = goal_info

        if self.category in ["correction", "mixed"]:
            output_dic["robot_detection"] = robot_detection_list   
            # if goal_reached and not robot_detection_list:
            #     output_dic["robot_detection"] = []
        else:
            pass
            # output_dic["robot_detection"] = []
        
        if not first_turn:
            output_dic["is_current_goal_reached"] = goal_reached
            output_dic["is_max_trial_reached"] = maxtry_reached
            
        if first_turn and "robot_detection" in output_dic:
            del output_dic["robot_detection"]

        if output_dic["current_goal"] == {}:
            output_dic["next_goal"] = goal_info
        else:
            output_dic["next_goal"] = {}

        # neaby_obj_info = []
        # if ins.nearby_obj and self.category == "mixed":
        #     for item in ins.nearby_obj:
        #         item_ins = self.topo_graph.instance_dict[item]
        #         obj_info = {
        #             "object_id": item_ins.id,
        #         }
        #         if item_ins.object_desc:
        #             obj_info["description"] = "|".join(item_ins.object_desc)
        #         if item_ins.attr:
        #             obj_info["special_attr"] = item_ins.attr
        #         neaby_obj_info.append(obj_info)
        # output_dic["nearby_objects"] = neaby_obj_info

        return_msg += json.dumps(output_dic, indent=2)
        return return_msg

    def _eval_with_agt_predict(
        self,
        current_goal: UserGoal,
        agtpose: Agent2DPose,
        semantic_img: np.ndarray,
        agt_predict: CountourMaskPrediction,
    ):
        # return robot detection list
        goal_contain_list = []
        return_list = []
        if agt_predict.flag == CountourMaskPrediction.CONTOUR_ONLY:
            masks = [None for _ in range(len(agt_predict.predict_contours))]
            contours = agt_predict.predict_contours
        else:
            masks = agt_predict.predict_masks
            contours = agt_predict.predict_contours

        for mask, contour in zip(masks, contours):
            is_goal, reach_oid_or_name = self._compare_detected_results(
                current_goal=current_goal,
                contour=contour,
                semantic_img=semantic_img,
                mask=mask,
            )
            if reach_oid_or_name is None:
                continue
            ins = self.topo_graph.instance_dict[current_goal.goal]
            dist, angle, _ = self.rel_pose(ins, agtpose)
            if reach_oid_or_name in [re.sub("_\d+", "", g) for g in ins.same_goal] and\
                dist < 30 and -45<angle<45:
                is_goal = True
            
            dic = {}
            if reach_oid_or_name in self.topo_graph.instance_dict:
                ins = self.topo_graph.instance_dict[reach_oid_or_name]
                dic["object_id"] = reach_oid_or_name
                dic["object_name"] = ins.name
                # if ins.room_id:
                #     dic["room_id"] = ins.room_id
                if ins.room_desc:
                    dic["room_name"] = ins.room_desc
                dic["is_goal"] = is_goal
                # if ins.object_desc and self.category in ["description", "mixed"]:
                #     dic["description"] = " | ".join(ins.object_desc)
                # if ins.attr and self.category in ["correction", "mixed"]:
                #     dic["special_attr"] = ins.attr
            else:
                dic["object_name"] = reach_oid_or_name
                if reach_oid_or_name == re.sub(r"\d+", "", current_goal.goal):
                    dic["is_goal"] = None  # not sure
                else:
                    dic["is_goal"] = is_goal
            ego_view_info = self._get_egoview_info_for_contour(contour, agtpose)
            if ego_view_info != "":
                dic["ego_view_info"] = ego_view_info
            
            if contour is not None:
                _, angle, _ = self.rel_pose(contour, agtpose)
            else:
                angle = 0

            goal_contain_list.append(is_goal)
            return_list.append((angle, dic))

        return_list = sorted(return_list, key=lambda x: x[0])

        return any(goal_contain_list), [item[1] for item in return_list]

    def _compare_detected_results(
        self,
        current_goal: UserGoal,
        contour: Optional[np.ndarray],
        semantic_img: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[bool, str]:
        # return is_goal, object_id or object_name
        logger.info(
            "[User Simulator] compare the ego-view detection results with ground truth"
        )
        goal_instance = self.topo_graph.instance_dict[current_goal.goal]
        goal_name = goal_instance.name

        if mask is not None:
            ego_view_detect_gtname = self._parse_detected_mask(semantic_img, mask, goal_name)
        else:
            ego_view_detect_gtname = None

        logger.info(
            f"[User Simulator] goal_name: {goal_name}, {current_goal.same_goal}, ego_view_detect_gtname: {ego_view_detect_gtname}"
        )
        # find the overlap between predicted contour and objects in the topograph

        # get possible instance to compare.
        if len(self.topo_graph.retrieve_objects_by_name(goal_name)) == 1 and ego_view_detect_gtname == goal_name:
            return True, goal_instance.id

        # print("retrieve_objects_by_name", self.topo_graph.retrieve_objects_by_name(goal_name))
        logger.info(f"[User Simulator] goal_name: {goal_name}")
        logger.info(f"[User Simulator] retrieve_objects_by_name: {self.topo_graph.retrieve_objects_by_name(goal_name)}")
        for oid in self.topo_graph.retrieve_objects_by_name(goal_name):
            ins = self.topo_graph.all_instance_dict[oid]
            if ins.contour is not None and contour is not None:
                overlap = vis.contour_overlap(
                    self.map_querier.mapcfg.num_grid, ins.contour, contour
                )
                logger.info(f"[User Simulator] overlap: {overlap}")
                if overlap > 0.8:
                    if ins.id in current_goal.same_goal:
                        return True, goal_instance.id  # find the goal,
                    else:
                        return (
                            False,
                            oid,
                        )  # find the same type object, but wrong instance

        logger.info(f"[User Simulator] ego_view_detect_gtname: {ego_view_detect_gtname}")
        if ego_view_detect_gtname is not None and ego_view_detect_gtname != goal_name:
            logger.info(f"[User Simulator] retrieve_objects_by_name: {self.topo_graph.retrieve_objects_by_name(ego_view_detect_gtname)}")
            for oid in self.topo_graph.retrieve_objects_by_name(ego_view_detect_gtname):
                ins = self.topo_graph.all_instance_dict[oid]
                if ins.contour is not None and contour is not None:
                    overlap = vis.contour_overlap(
                        self.map_querier.mapcfg.num_grid, ins.contour, contour
                    )
                    logger.info(f"[User Simulator] overlap: {overlap}")
                    if overlap > 0.8:
                        return False, oid  # find the wrong type object
        
        for o, ins in self.topo_graph.instance_dict.items():
            # find other possible nearest object
            if ins.name in [goal_name, ego_view_detect_gtname]:
                continue

            # compare distance
            dist = self.topo_graph.point_contour_dist(
                p=(self.agtpose.x, self.agtpose.z), c=ins.contour
            )

            if dist > self.goal_radius: continue

            if ins.contour is not None and contour is not None:
                overlap = vis.contour_overlap(self.map_querier.mapcfg.num_grid, ins.contour, contour)
                if overlap > 0.92:
                    return False, o

        return False, ego_view_detect_gtname

    def _parse_detected_mask(self, semantic_img, mask, goal_name=None):
        if len(semantic_img.shape) == 3:
            semantic_img = semantic_img.squeeze()
        if len(mask.shape) == 3:
            mask = mask.squeeze()
        assert semantic_img.shape == mask.shape
        predict_semantics = semantic_img[mask]
        # find the largest semantic class
        bin_count = np.bincount(predict_semantics)
        if len(bin_count) > 3:
            bin_count[0] = 0  # ignore background
            bin_count[1] = 0  # ignore wall
            bin_count[2] = 0  # ignore floor
            largest_semantic_id = np.argmax(bin_count)
            logger.info(
                f"[User Simulator] largest_semantic_id found in masked semantic: {largest_semantic_id}"
            )
            if largest_semantic_id in [0, 1, 2] or np.max(bin_count) < 10:
                return None
            if largest_semantic_id in self.label_dic:
                clsname = self.label_dic[largest_semantic_id]
                logger.info(f"[User Simulator] clsname: {clsname}")
            else:
                return "none"

            if goal_name is not None:
                # get top 3 semantic class
                top3_semantic_id = np.argsort(bin_count)[-3:]
                logger.info(f"[User Simulator] check top 3 semantic class: {top3_semantic_id}")
                for sid in top3_semantic_id:
                    if sid in [0, 1, 2]:
                        continue
                    if sid in self.label_dic:
                        logger.info(f"[User Simulator] clsname: {self.label_dic[sid]}")
                        if self.label_dic[sid] == goal_name:
                            return goal_name
            return clsname
        else:
            return "wall"

    @staticmethod
    def _post_process(response_str) -> str:
        try:
            dic = json.loads(response_str)
            response = dic["Response"]
        except Exception as e:
            m = re.search(r"\"Response\"\: \"(.*?)\"", response_str)
            if m:
                response = m.group(1)
            else:
                response = response_str
        return response