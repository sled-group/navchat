from copy import deepcopy
import json
import re
from typing import List, Optional, Tuple
from attr import define
import numpy as np

from orion.config.my_config import *
from orion.navigation.shortest_path_follower_wrapper import MyFollower
from orion.navigation.fmm_planner import MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
from orion.user_simulator.topograph import TopologicalGraph
from orion.abstract.pose import Agent2DPose

import random

random.seed(1)


@define
class UserGoal:
    goal: str  # object id
    round: int = 1  # current round
    trial: int = 0  # current trial
    success: bool = False
    steps: int = 0
    reference_steps: int = 0
    same_goal: Optional[
        Tuple[str]
    ] = None  # tuple of object id. reach any of it consider success

    def __str__(self):
        return f"goal: {self.goal}, round: {self.round}, trial: {self.trial}, success: {self.success}, steps: {self.steps}, goal_objs: {self.same_goal}"

    def __repr__(self):
        return self.__str__()


class UserGoalGenerator:
    def __init__(
        self,
        topograph: TopologicalGraph,
        max_trial: int = 5,
        repeat_round: int = 3,
        follower: MyFollower = MyFollower(),
        init_agtpose2d=Agent2DPose(
            MapConfig().num_grid // 2, MapConfig().num_grid // 2, -np.pi / 2
        ),
    ):
        self.topograph = topograph
        self.max_trial = max_trial
        self.repeat_round = repeat_round
        self.follower = follower
        self.init_agtpose2d = init_agtpose2d
        self.last_round = 1
        self._is_new_round = False

    def reset(self):
        self.goals: List[UserGoal] = []
        goal_obj_ids = deepcopy(list(self.topograph.instance_dict.keys()))
        # goal_obj_ids = ["bed_0", "couch_0"] # for debug
        for i in range(self.repeat_round):
            random.shuffle(goal_obj_ids)
            for g in goal_obj_ids:
                self.goals.append(
                    UserGoal(
                        goal=g,
                        round=i + 1,
                        same_goal=self.topograph.instance_dict[g].same_goal
                    )
                )
        self.goal_idx = 0
        reference_step_number, _ = self._get_reference_movements(
            self.goals[self.goal_idx], self.init_agtpose2d
        )
        self.goals[self.goal_idx].reference_steps = reference_step_number

    def step(self, success: bool, steps: int, first_turn: bool = False, agtpose=None):
        if not first_turn:
            self.current_goal.trial += 1
            self.current_goal.steps += steps
            self.current_goal.success = success

        maxtry_reached = self.maxtry_reached
        task_finished = self.task_finished

        if success or maxtry_reached:
            self.goal_idx += 1
            if self.goal_idx < len(self.goals):
                reference_step_number, _ = self._get_reference_movements(
                    self.goals[self.goal_idx], agtpose
                )
                self.goals[self.goal_idx].reference_steps = reference_step_number
                if self.last_round == self.current_goal.round:
                    self._is_new_round = False
                else:
                    self._is_new_round = True
                    self.last_round = self.current_goal.round

        return maxtry_reached, task_finished

    def _get_reference_movements(self, goal: UserGoal, agtpose: Agent2DPose):
        goal_ins = self.topograph.instance_dict[goal.goal]
        goal_center = goal_ins.center
        # find the closest node to the goal in navigable area
        mask = self.follower.get_navigable_mask().copy()
        closest_point_goal = self._get_cloest_pt(mask, goal_center[0], goal_center[1])
        closest_point_start = self._get_cloest_pt(mask, agtpose.x, agtpose.z)
        reachable, _, a_list = self.follower.follower.plan(
            start=Agent2DPose(
                closest_point_start[0], closest_point_start[1], agtpose.t
            ),
            goal=Agent2DPose(closest_point_goal[0], closest_point_goal[1]),
            plot=False,
        )
        if reachable:
            return len(a_list), self._get_instruct_string(a_list)
        else:
            return 10, ""

    @staticmethod
    def _get_cloest_pt(mask, x, y):
        rows, cols = np.where(mask)
        distances = np.sqrt((rows - y) ** 2 + (cols - x) ** 2)
        closest_idx = np.argmin(distances)
        closest_point = (cols[closest_idx], rows[closest_idx])
        return closest_point

    @staticmethod
    def _get_instruct_string(a_list):
        action_string = ""
        for a in a_list:
            if a == MOVE_FORWARD:
                action_string += "f"
            elif a == TURN_LEFT:
                action_string += "l"
            elif a == TURN_RIGHT:
                action_string += "r"
        action_str_list = []
        for step in re.split(r"(lll+|rrr+|fff+)", action_string):
            accu_angle = 0
            accu_dist = 0
            for s in step:
                if s == "l":
                    accu_angle -= 15
                elif s == "r":
                    accu_angle += 15
                elif s == "f":
                    accu_dist += 1

            if abs(accu_angle) >= 135:
                act = "turn around"
            elif accu_angle <= -45:
                if accu_dist >= 3:
                    act = f"turn left and go forward {accu_dist*5} units"
                else:
                    act = "turn left"
            elif accu_angle >= 45:
                if accu_dist >= 3:
                    act = f"turn right and go forward {accu_dist*5} units"
                else:
                    act = "turn right"
            elif accu_dist >= 3:
                act = f"go forward {accu_dist*5} units"
            else:
                act = ""
            if act == "":
                continue
            if len(action_str_list) > 0:
                if action_str_list[-1] in [
                    "turn left",
                    "turn right",
                    "turn around",
                ] and act in ["turn left", "turn right", "turn around"]:
                    if len(action_str_list) == 1:
                        action_str_list[-1] = "turn around"
                    else:
                        pass
                elif action_str_list[-1].startswith("go forward") and act.startswith(
                    "go forward"
                ):
                    last_num = action_str_list[-1].split(" ")[-2]
                    num = act.split(" ")[-2]
                    action_str_list[-1] = f"go forward {int(last_num)+int(num)} units"
                else:
                    action_str_list.append(act)
            else:
                action_str_list.append(act)
        return [a for a in action_str_list if a != ""]

    @property
    def last_goal(self):
        if self.goal_idx == 0:
            return None
        return self.goals[self.goal_idx]

    @property
    def current_goal(self):
        if self.goal_idx >= len(self.goals):
            return None
        return self.goals[self.goal_idx]

    @property
    def next_goal(self):
        if self.goal_idx + 1 >= len(self.goals):
            return None
        return self.goals[self.goal_idx + 1]

    @property
    def maxtry_reached(self):
        return self.current_goal.trial >= self.max_trial

    @property
    def task_finished(self):
        if self.current_goal is None:
            return True
        elif (
            self.current_goal.success or self.maxtry_reached
        ) and self.next_goal is None:
            return True

        return False

    def save(self, path):
        out_put = {}
        for round in range(1, self.repeat_round + 1):
            dic_one_round = {}
            for goal in self.goals:
                if goal.round == round:
                    dic_one_round[goal.goal] = {
                        "success": goal.success,
                        "steps": goal.steps,
                        "reference_steps": goal.reference_steps,
                        "trial": goal.trial,
                    }
            out_put[f"round_{round}"] = dic_one_round
        json.dump(out_put, open(path, "w"), indent=2)
        return out_put


    def evaluate(self):
        result = {}
        for round in range(1, self.repeat_round + 1):
            result[round] = self._measure(round=round)
        result["all"] = self._measure()
        return result