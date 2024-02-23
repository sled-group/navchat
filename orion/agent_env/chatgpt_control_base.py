"""
hybrid search + chatgpt issue command + talk with user

"""

import json
import os
import re
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np

from orion import logger
from orion.abstract.interaction_history import (
    BotMsg,
    FuncMsg,
    GPTMsg,
    InteractionHistory,
    StepAct,
    UsrMsg,
)
from orion.abstract.pose import Agent2DPose, ObjectWayPoint
from orion.agent_env.habitat.utils import try_action_import_v2
from orion.agent_env.hybrid_search import HybridSearchBase
from orion.chatgpt.prompts.agent_prompts import SYSTEM_PROMPT
from orion.chatgpt.api import ChatAPI
from orion.chatgpt.prompts.agent_functions import FUNCTIONS
from orion.config.chatgpt_config import *
from orion.config.my_config import *

MOVE_FORWARD, _, TURN_LEFT, TURN_RIGHT, STOP = try_action_import_v2()
action_dict = {
    "move_forward": MOVE_FORWARD,
    "turn_left": TURN_LEFT,
    "turn_right": TURN_RIGHT,
    "stop": STOP,
}
inverse_action_dict = {v: k for k, v in action_dict.items()}


class ChatGPTControlBase(HybridSearchBase):
    """Do not use the function call version"""

    def __init__(
        self,
        chatgpt_config,
        dump_dir="dump",
        use_stream=False,
        record_interaction=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_stream = use_stream
        self.dump_dir = os.path.join(self.save_dir, dump_dir)

        # This record is for future model-training, like Decision Transformer
        # Will record pose, rgb, dialog, actions, etc.
        self.interaction_history = InteractionHistory(record=record_interaction)
        self.record_conversations = []  # This is only record dialog

        self.chatgpt = ChatAPI(config=chatgpt_config)
        self.chatgpt.message = [{"role": "system", "content": SYSTEM_PROMPT}]

        self.object_id_dict: Dict[
            str, ObjectWayPoint
        ] = dict()  # {object_id: object_waypt}

        self._clear_prediction_for_user_simulator()

        self.reached_vlmap_oids = []
        self.large_object_contours = []  # avoid mis-detection
        self.tmp_masked_objwpts = []

    def step(self, action):
        self._clear_prediction_for_user_simulator()
        super().step(action)
        if self.clip_text_list:
            self._detect_with_clip(self.observations.rgb)
            logger.info(f"clip prob {self.clip_scorer.prob_list[-1]}")

        self.interaction_history.append(
            StepAct(
                action=inverse_action_dict[action],
                next_obs=self.observations,
                next_state=self.agent_state,
            )
        )

    def _get_reference_movements(self, center: Tuple[int, int], agtpose: Agent2DPose):
        goal_center = center
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
            return max(len(a_list), 5)
        else:
            return 10

    @staticmethod
    def _get_cloest_pt(mask, x, y):
        rows, cols = np.where(mask)
        distances = np.sqrt((rows - y) ** 2 + (cols - x) ** 2)
        closest_idx = np.argmin(distances)
        closest_point = (cols[closest_idx], rows[closest_idx])
        return closest_point

    def loop(self):
        self.task_finished = False
        self.interaction_history.append(
            StepAct(
                action=None, next_obs=self.observations, next_state=self.agent_state
            )
        )

        while True:
            utterance = self._get_user_input()
            if utterance == "exit":
                break

            self.record_conversations.append(
                {
                    "role": "user",
                    "content": utterance,
                }
            )

            if self.task_finished:
                logger.info("All Task Finished")
                self.interaction_history.append(UsrMsg(text=utterance))
                break

            self.chatgpt.add_user_message(utterance)
            self.interaction_history.append(UsrMsg(text=utterance))
            logger.info(f"\n[User Utterance] {utterance}")

            response = self._get_chatgpt_response()
            logger.info(f"[GPT Response] {response}")
            self.chatgpt.add_assistant_message(response)
            command = self._resp2cmd(response)

            time.sleep(2)
            self._save_money_gpt_trucate()

            if command is None:
                logger.warning("GPT Response Command is None")
                self._post_process(command)
                continue

            while command["name"] != "dialog":
                try:
                    return_msg = self.execute(command)
                except Exception as e:
                    logger.warning(e)
                    return_msg = e
                    traceback.print_exc()

                if return_msg is not None:
                    self._send_funcall_msg(return_msg)
                    response = self._get_chatgpt_response()

                    logger.info(f"[GPT Response] {response}")
                    while response == "":
                        logger.info(
                            "GPT Response is empty, may due to internet connection, wait serval seconds"
                        )
                        time.sleep(2)
                        response = self._get_chatgpt_response()

                    self.chatgpt.add_assistant_message(response)
                    command = self._resp2cmd(response)
                    time.sleep(2)
                    self._save_money_gpt_trucate()

                    if command is None:
                        logger.warning("GPT Response Command is None")
                        break

            if command is not None and command["name"] == "dialog":
                self.interaction_history.append(BotMsg(text=command["args"]["content"]))

                self.record_conversations.append(
                    {
                        "role": "robot",
                        "content": command["args"]["content"],
                    }
                )

            self._post_process(command)

    def _save_money_gpt_trucate(self):
        if (
            re.search(r"(gpt-4|gpt4)", self.chatgpt.model)
            and self.chatgpt.usage_tokens > 14000
            and not self.chatgpt.do_truncation
        ):
            self.chatgpt.truncate(percentage=2)

    def _get_user_input(self):
        return input("User>>>")

    def _send_funcall_msg(self, msg):
        logger.info(f"[Function Return] {msg}")
        self.chatgpt.add_function_message(msg)

    def _get_chatgpt_response(self):
        try:
            if self.use_stream:
                response: str = ""
                for chunk in self.chatgpt.get_system_response_stream():
                    response += chunk
            else:
                response = self.chatgpt.get_system_response()
        except Exception as e:
            logger.warning(f"When getting response from gpt: {e}")
            traceback.print_exc()
            response = ""
        return response

    def execute(self, command) -> str:
        raise NotImplementedError

    def _post_process(self, command):
        if command is None:
            self.agent_response = "I may make a mistake, please provide more information, let me try again"
        else:
            self.agent_response = command["args"]["content"]

    def _resp2cmd(self, response_message: str) -> Optional[Dict[str, Any]]:
        """parse response message to command dict
        sometimes the response message is not a valid json string,
        so we need to extract the command from it
        """
        if not response_message.endswith("\n}"):
            response_message = response_message + "\n}"
        if "Command" not in response_message and "Thought" not in response_message:
            return {"name": "dialog", "args": {"content": response_message}}
        try:
            resp_dic = json.loads(response_message.strip())
            command: Optional[Dict[str, Any]] = resp_dic["Command"]
        except:
            command = None

        if command is not None:
            return command

        m = re.search(r"(\{\"name\"\:.*\}\})", response_message)
        try:
            command = json.loads(m.group(1).strip())  # type: ignore
        except:
            command = None
        return command

    def _update_prediction_for_user_simulator(self, mbbox, contour):
        logger.info("update prediction for user simulator")
        if (
            mbbox is not None
            and contour is not None
            and isinstance(contour, list)
            and len(mbbox) == len(contour)
        ):
            # egoview detect with multiple detections
            self.predict_masks = [m.squeeze() for m in mbbox.masks]
            self.predict_contours = contour
            logger.info(
                f"{len(self.predict_masks)} masks, {len(self.predict_contours)} contours"
            )

        elif mbbox is not None and isinstance(contour, np.ndarray) and len(mbbox) == 1:
            # egoview detect with only one detection
            self.predict_masks = [mbbox.masks[0].squeeze()]
            self.predict_contours = [contour]
            logger.info(
                f"{len(self.predict_masks)} masks, {len(self.predict_contours)} contours"
            )

        elif contour is not None and isinstance(contour, np.ndarray):
            # retrieve from memory, only contour
            self.predict_masks = None
            self.predict_contours = [contour]
            logger.info(f"contour only {len(self.predict_contours)} contours")

    def _clear_prediction_for_user_simulator(self):
        self.predict_masks = None
        self.predict_contours = None