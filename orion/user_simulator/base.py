import json
import pickle
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from orion.abstract.pose import Agent2DPose, get_rotation_angle2d


from orion.config.my_config import *
from orion.map.map_search.search_voxel import MapSearch
from orion.utils.file_load import (
    cvt_sem_id_2_cls_id,
    load_obj2cls_dict,
    get_floor_set_str,
)
from orion.navigation.waypoint_planner import PointPlanner
from orion import logger
from orion.user_simulator.topograph import TopologicalGraph, Instance
from orion.user_simulator.user_goal import UserGoal, UserGoalGenerator
from orion.chatgpt.api import ChatAPI
from orion.config.chatgpt_config import *
from orion.abstract.usersim import UserSimulator
from orion.navigation.shortest_path_follower_wrapper import MyFollower

from orion.chatgpt.prompts.usersim_prompts_mix import SYSTEM_PROMPT as SYSTEM_PROMPT_mixed
from orion.chatgpt.prompts.usersim_prompts_instruction import SYSTEM_PROMPT as SYSTEM_PROMPT_instruction
from orion.chatgpt.prompts.usersim_prompts_description import SYSTEM_PROMPT as SYSTEM_PROMPT_description
from orion.chatgpt.prompts.usersim_prompts_correction import SYSTEM_PROMPT as SYSTEM_PROMPT_correction
from orion.chatgpt.prompts.usersim_prompts_landmark import SYSTEM_PROMPT as SYSTEM_PROMPT_landmark
from orion.chatgpt.prompts.usersim_prompts_none import SYSTEM_PROMPT as SYSTEM_PROMPT_none


import random

random.seed(1)


class UserSimulatorBase(UserSimulator):
    def __init__(
        self,
        scene_id,
        floor_plan,
        chatgpt_usrsim_config=AzureGPT35Config(), # GPT3.5 is enough for user simulator
        max_trial=5,
        max_round=5,
        category="mixed",
        goal_radius=20,
        clear_gptcontext=False,
        is_cow_baseline=False,
        is_vlamp_baseline=False,
    ):
        print("is_cow_baseline", is_cow_baseline)
        print("is_vlamp_baseline", is_vlamp_baseline)

        self.scene_id = scene_id
        self.floor_plan = floor_plan
        self.clear_gptcontext = clear_gptcontext
        logger.info(
            f"start user simulator. scene_id: {scene_id}, floor_plan: {floor_plan}"
        )
        load_sparse_map_path = f"data/experiments/predict_{scene_id}_{get_floor_set_str(floor_plan)}/lseg_vlmap/sparse_vxl_map.npz"
        self.map_querier = MapSearch(load_sparse_map_path=load_sparse_map_path)

        semantic_dict_path = f"data/experiments/predict_{scene_id}_{get_floor_set_str(floor_plan)}/recordings/obj2cls_dict.txt"
        self.obj2cls_dic, self.label_dic = load_obj2cls_dict(semantic_dict_path)
        self.name_dic = {
            v[1]: (k, v[0]) for k, v in self.obj2cls_dic.items()
        }  # {name: (semantic_id, cls_id)}

        self.topo_graph = TopologicalGraph(
            load_sparse_map_path, self.label_dic, self.map_querier
        )

        gt_mask_path = f"data/experiments/predict_{scene_id}_{get_floor_set_str(floor_plan)}/gt_navigable_mask.npy"
        self.follower = MyFollower()
        self.follower.set_traversible_map(np.load(gt_mask_path))

        self.agtpose = None
        self.last_step_count = 0

        self.max_trial = max_trial  # for one goal, max trial times
        self.max_round = max_round  # repeat times for total goals, can be used to draw the learning curve
        self.category = category  # ["correction", "description", "instruction", "landmark", "mixed"]
        assert self.category in [
            "correction",
            "description",
            "instruction",
            "landmark",
            "mixed",
            "none",
        ], f"category {category} not supported"

        if self.category in ["mixed"]:
            logger.info("loading prompts from navchat.chatgpt.usersim_prompts_mixed")
            self.SYSTEM_PROMPT = SYSTEM_PROMPT_mixed
        elif self.category in ["instruction"]:
            logger.info("loading prompts from navchat.chatgpt.usersim_prompts_instruction")
            self.SYSTEM_PROMPT = SYSTEM_PROMPT_instruction
        elif self.category in ["description"]:
            logger.info("loading prompts from navchat.chatgpt.usersim_prompts_description")
            self.SYSTEM_PROMPT = SYSTEM_PROMPT_description
        elif self.category in ["correction"]:
            logger.info("loading prompts from navchat.chatgpt.usersim_prompts_correction")
            self.SYSTEM_PROMPT = SYSTEM_PROMPT_correction
        elif self.category in ["landmark"]:
            logger.info("loading prompts from navchat.chatgpt.usersim_prompts_landmark")
            self.SYSTEM_PROMPT = SYSTEM_PROMPT_landmark
        elif self.category in ["none"]:
            logger.info("loading prompts from navchat.chatgpt.usersim_prompts_none")
            self.SYSTEM_PROMPT = SYSTEM_PROMPT_none
        else:
            raise ValueError(f"category {category} not supported")


        self.goal_radius = goal_radius  # 10 units = 2.5 meters

        # We use gpt3.5 turbo, it should be enough for the user simulator
        self.chatgpt = ChatAPI(config=chatgpt_usrsim_config)
        self.reset()
        self.is_cow_baseline = is_cow_baseline

    def generate_hint(self, instance: Instance) -> str:
        # from semantic map.  neighbor objects in view.  largest object in circle 15. randomly

        dist, angle, is_in_view = self.rel_pose(instance)

        # if far, return object hint
        if dist > 100 or not is_in_view:
            # return f"the {instance.name} is {dist} meters away from you"
            nearby_objs = instance.nearby_obj
            if len(nearby_objs) > 0:
                nearby_obj = nearby_objs.pop(0)
                if nearby_obj.name == instance.name:
                    return f"the {instance.name} is near to another {nearby_obj.name}"
                else:
                    return f"the {instance.name} is near to a {nearby_obj.name}"
        # if close, return postion hint
        else:
            if -30 < angle <= 30:
                return f"the {instance.name} is in front of you around {dist} units"
            elif 30 < angle <= 60:
                return f"the {instance.name} is in front of you and at your right side around {dist} units"
            elif 60 < angle <= -120:
                return f"the {instance.name} is at your right side around {dist} units"
            elif 120 < angle <= 150:
                return f"the {instance.name} is behind you and at your right side around {dist} units"
            elif 150 < angle <= 180 or -180 <= angle <= -150:
                return f"the {instance.name} is behind you around {dist} units"
            elif -150 < angle <= -120:
                return f"the {instance.name} is behind you and at your left side around {dist} units"
            elif -120 < angle <= -60:
                return f"the {instance.name} is at your left side around {dist} units"
            elif -60 < angle <= -30:
                return f"the {instance.name} is in front of you and at your left side around {dist} units"

    def rel_pose(
        self,
        object_contour: Union[np.ndarray, Instance],
        agtpose: Agent2DPose,
    ) -> Tuple[float, float, bool]:
        # given current pose, return the relative position of the instance <distance, angle>
        if isinstance(object_contour, Instance):
            contour = object_contour.contour
            center = object_contour.center
        else:
            contour = object_contour
            M = cv2.moments(contour)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        _, is_in_view = PointPlanner.line_search(
            x0=center[0],
            y0=center[1],
            x1=agtpose.x,
            y1=agtpose.z,
            wall_mask=self.topo_graph.wall_mask,
            stop_at_wall=True,
            object_contours=contour,
        )
        x, z = agtpose.x, agtpose.z

        dist = self.topo_graph.point_contour_dist((x, z), contour)

        angle = get_rotation_angle2d(
            src_pose=agtpose,
            tgt_pose=Agent2DPose(x=center[0], z=center[1]),
        )

        return dist, angle, is_in_view

    def _cvt_semid2clsid(self, semantic: np.ndarray):
        # raw semantic id image -> cls id
        if len(semantic.shape) == 3:
            semantic = semantic.squeeze()
        semantic = np.asarray(semantic).astype(np.int32)
        semantic = cvt_sem_id_2_cls_id(semantic, self.obj2cls_dic)
        return semantic

    def _is_in_egoview(self, semantic_image: np.ndarray, instance: Instance):
        object_name = instance.name

        if object_name not in self.name_dic:
            logger.warning(f"{object_name} not in the name_dic")
            return False

        oid = self.name_dic[object_name][1]

        mask = semantic_image == oid
        if np.sum(mask) == 0:
            return False

        # center of the mask
        y, x = np.where(mask)
        ymin, ymax = np.min(y), np.max(y)
        xmin, xmax = np.min(x), np.max(x)

        x_center = (xmax + xmin) // 2
        H, W = semantic_image.shape

        if instance.mass < 100:
            thres = 50
        elif instance.mass < 500:
            thres = 150
        else:
            thres = 350

        if (
            np.sum(mask) > thres and 0.3 * W < x_center < 0.7 * W
        ):  # can not be too small or too aside
            return True
        else:
            return False

    def reset(self):
        usr_goal_file = f"orion/user_simulator/goals/{self.scene_id}/final.json"
        all_object_file = f"orion/user_simulator/goals/{self.scene_id}/objects.json"
        # load from files
        self.topo_graph.load_instance_data(
            json.load(open(usr_goal_file)), json.load(open(all_object_file))
        )

        self.goal_gen = UserGoalGenerator(
            self.topo_graph,
            max_trial=self.max_trial,
            repeat_round=self.max_round,
            follower=self.follower,
        )
        self.goal_gen.reset()
        self.chatgpt.message = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]

    def evaluate(self):
        return self.goal_gen.evaluate()

    def step(
        self,
        agent_response: str,
        semantic_img: np.ndarray,
        agtpose: Agent2DPose,
        step_count: int,
    ) -> Tuple[bool, str]:
        # return task_finished, response
        return NotImplementedError

    def _eval_with_semantic_img(
        self, goal: UserGoal, agtpose: Agent2DPose, semantic_img: np.ndarray
    ) -> bool:
        # This could potentially be false postive.
        # But should be adequet for navigation task
        goal_objs = goal.same_goal
        goal_reached = False

        for obj in goal_objs:
            ins = self.topo_graph.all_instance_dict[obj]
            dist, angle, is_in_view = self.rel_pose(ins, agtpose)
            logger.info(f"[User Simulator Measure] {dist}, {angle}, {is_in_view}" )
            if dist < 12 and is_in_view and -45 < angle < 45:
                goal_reached = True
            elif dist < 24 and is_in_view and -30 < angle < 30:
                goal_reached = True
            elif (
                dist < 28 and self._is_in_egoview(semantic_img, ins)
            ):
                goal_reached = True
            
            if self.is_cow_baseline:
                logger.info(f"[User Simulator Measure] COW baseline")
                if dist < 40 and is_in_view and -90 < angle < 90:
                    goal_reached = True
        return goal_reached

    @staticmethod
    def _wrap_number(number, multiple):
        num = int(number)
        if num>0:
            return num // multiple * multiple
        else:
            return -(-num // multiple) * multiple

    def _get_egoview_info_for_goal(self, goal: UserGoal, agtpose: Agent2DPose) -> str:
        goal_objs = goal.same_goal
        tuple_list = []
        for obj in goal_objs:
            ins = self.topo_graph.all_instance_dict[obj]
            dist, angle, is_in_view = self.rel_pose(ins, agtpose)
            tuple_list.append((obj, dist, angle, is_in_view))
        egoview_info = sorted(tuple_list, key=lambda x: x[0])[0]
              
        units =  self._wrap_number(egoview_info[1], 15)
        degrees = self._wrap_number(egoview_info[2], 30)

        if -15<=degrees<=15:
            desc_str = "(in the front)"
        elif 15<degrees<=165:
            desc_str = "(in the right)"
        elif 165<degrees<=180 or -180<=degrees<=-165:
            desc_str = "(at the back)"
        elif -165<=degrees<-15:
            desc_str = "(in the left)"
        else:
            desc_str = ""

        tuple_str = f"distance: {units} units, angle: {degrees} degrees {desc_str}, in_same_room: {egoview_info[3]}"

        if egoview_info[3]:
            return {"tuple_str": tuple_str, "route": []}
        else:
            _, desc = self.goal_gen._get_reference_movements(goal, agtpose)
            return {"tuple_str": tuple_str, "route": desc}

    def _get_egoview_info_for_contour(
        self, contour: np.ndarray, agtpose: Agent2DPose
    ) -> str:
        if contour is None:
            return ""
        dist, angle, is_in_view = self.rel_pose(contour, agtpose)
        # return dist, angle, is_in_view
        return f"distance: {int(dist)} units, angle: {int(angle)} degrees, in_same_room: {is_in_view}"
    

    def eval(self, results):
        for round in results:
            logger.info(f"\nRound {round}:")
            self.measure(results[round])

    def measure(self, goals):
        success_list = []
        spl_list = []
        sit_list = []

        small_obj_list = []
        big_obj_list = []
        ambiguous_obj_list = []

        for oid, item in goals.items():
            if item["reference_steps"] == 0:
                logger.info(f"{oid} not started")
                continue

            success = item["success"]
            steps = item["steps"]
            reference_steps = item["reference_steps"]
            trial = item["trial"]
            if trial == 0: continue
            spl = success * reference_steps / max(steps, reference_steps)
            sit = success/trial
            success_list.append(success)
            spl_list.append(spl)
            sit_list.append(sit)
            logger.info(f"{oid} success: {success}, spl: {spl:2f}, sit: {sit:2f}")

            typ = self.topo_graph.instance_dict[oid].type
            if typ == "small":
                small_obj_list.append(success)
            elif typ == "big":
                big_obj_list.append(success)
            else:
                ambiguous_obj_list.append(success)
    
        if len(success_list) > 0:
            logger.info(f"Success: {float(np.mean(success_list)):2f}, spl: {float(np.mean(spl_list)):2f}, sit: {float(np.mean(sit_list)):2f}")
        else:
            logger.info(f"Success: NaN, spl: NaN, sit: NaN")
        
        if len(small_obj_list) > 0:
            logger.info(f"small objects: {float(np.mean(small_obj_list)):2f}")
        else:
            logger.info(f"small objects: NaN")
        
        if len(big_obj_list) > 0:
            logger.info(f"big objects: {float(np.mean(big_obj_list)):2f}")
        else:
            logger.info(f"big objects: NaN")

        if len(ambiguous_obj_list) > 0:
            logger.info(f"ambiguous objects: {float(np.mean(ambiguous_obj_list)):2f}")
        else:
            logger.info(f"ambiguous objects: NaN")

            