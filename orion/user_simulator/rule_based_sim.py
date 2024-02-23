"""
This file is deprecated.
"""

import re
from typing import List, Tuple
import numpy as np
from orion.abstract.pose import Agent2DPose


from orion.config.my_config import *
from orion.user_simulator.topograph import Instance
from orion.config.chatgpt_config import *
from orion.user_simulator.base import UserSimulatorBase

import random

random.seed(1)


class RuleUserSimulator(UserSimulatorBase):
    def generate_hint(self, instance: Instance) -> str:
        # from semantic map.  neighbor objects in view.  largest object in circle 15. randomly

        dist, angle, is_in_view = self.rel_pose(instance, self.agtpose)

        # if far, return object hint
        if dist > 100 or not is_in_view:
            # return f"the {instance.name} is {dist} meters away from you"
            nearby_objs = self.topo_graph.get_sorted_neighbors(instance.id)
            if len(nearby_objs) > 0:
                nearby_obj = nearby_objs.pop(0)
                nearby_obj_name = re.sub(r"_\d+$", "", nearby_obj)
                if nearby_obj_name == instance.name:
                    return f"the {instance.name} is near to another {nearby_obj_name}"
                else:
                    return f"the {instance.name} is near to a {nearby_obj_name}"
        # if close, return postion hint
        else:
            if -30 < angle <= 30:
                return f"the {instance.name} is in front of you around {dist} units"
            elif 30 < angle <= 60:
                return f"the {instance.name} is in front of you and at your right side around {dist} units"
            elif 60 < angle <= 120:
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

    def step(
        self,
        agent_response: str,
        semantic_img: np.ndarray,
        agtpose: Agent2DPose,
        step_count: int,
    ) -> Tuple[bool, str]:
        self.agtpose = agtpose

        goal_reached = self._eval_with_semantic_img(
            self.goal_gen.current_goal, agtpose, semantic_img
        )
        nearest_tuple = self._get_egoview_info_for_goal(
            self.goal_gen.current_goal, agtpose
        )

        maxtry_reached, task_finished = self.goal_gen.step(
            goal_reached, steps=step_count - self.last_step_count
        )
        self.last_step_count = step_count

        if task_finished:
            return True, "That's all for today. Thank you for your help."

        return_str = ""
        if step_count > 0:
            if goal_reached:
                return_str += f"Yes that's correct. "
            else:
                return_str += f"No, you're wrong. "

        goal = self.goal_gen.current_goal
        ins = self.topo_graph.instance_dict[goal.goal]
        if maxtry_reached or goal_reached or step_count == 0:
            return_str += f"Now I want you to find the {ins.name} "
            if ins.nearby_obj:
                near_obj_str = ", ".join(
                    [re.sub(r"_\d+$", "", obj) for obj in ins.nearby_obj]
                )
                return_str += f"which near to the {near_obj_str}. "

        else:
            hints = self.generate_hint(ins)
            if hints is not None:
                return_str += f" Hints: {hints}. "
            else:
                return_str += f" Please find the {ins.name}. "

        return task_finished, return_str


if __name__ == "__main__":
    for scene, floor in SCENE_ID_FLOOR_SET:
        usr_sim = RuleUserSimulator(scene, floor, max_round=2, category=2)
        for k, v in usr_sim.topo_graph.instance_dict.items():
            print(k, v)
        input()
        for ii in usr_sim.goal_gen.goals:
            print(ii)
        ctt = 0
        task_finished = False
        while True:
            user_input = input("user: ")
            if user_input == "next" or task_finished:
                break
            if (ctt + 1) % 3 == 0:
                # mock
                g = usr_sim.goal_gen.current_goal
                ins = usr_sim.topo_graph.instance_dict[g.goal]
                x, z = ins.center
                agtpose = Agent2DPose(x, z, 0)
                cls_id = usr_sim.name_dic[ins.name][0]
                semantic_img = np.ones((100, 100)) * cls_id
            else:
                semantic_img = np.zeros((100, 100))
                agtpose = Agent2DPose(0, 0, 0)

            task_finished, response = usr_sim.step(
                user_input, semantic_img, agtpose, ctt
            )
            print("bot: ", response)
            ctt += 1

        input("press enter to continue")
