import os
import pickle
from copy import deepcopy
from typing import List, Optional, Union

import cv2
import numpy as np
from attr import define

from orion import logger
from orion.abstract.agent import AgentState
from orion.abstract.interfaces import Observations
from orion.abstract.perception import MaskedBBOX
from orion.abstract.pose import Agent2DPose, Agent3DPose


@define
class UsrMsg:
    # usr utterance
    text: str


@define
class SucMsg:
    # success message
    reward: float


@define
class GPTMsg:
    # chatgpt response
    content: str


@define
class FuncMsg:
    # function return message
    content: str


@define
class BotMsg:
    # robot response
    text: str


@define
class PointAct:
    # waypoint
    pt2d: Agent2DPose
    pt3d: Agent3DPose  # pose by simulaotor [x, y, z]


@define
class StepAct:
    action: Optional[str]  # low-level action
    next_obs: Observations
    next_state: AgentState

    def compression(self):
        self.next_obs = None
        # # depth consumes too much memory
        # self.next_obs.semantic = None
        # self.next_obs.depth = (self.next_obs.depth * 1000).astype(np.uint16)
        # if self.next_obs.info is not None and "collisions" in self.next_obs.info and self.next_obs.info["collisions"]["is_collision"]:
        #     self.next_obs.info = {"collisions": self.next_obs.info["collisions"]}
        # else:
        #     self.next_obs.info = None


@define
class DetAct:
    mmbox: MaskedBBOX


JointType = Union[UsrMsg, GPTMsg, BotMsg, PointAct, StepAct, DetAct]


class InteractionHistory:
    def __init__(self, record=False):
        self.record = record
        self.interactions: List[JointType] = []

    def append(self, item: JointType):
        if not self.record:
            return
        _item = deepcopy(item)

        if isinstance(_item, StepAct):
            _item.compression()

        self.interactions.append(_item)

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        return self.interactions[idx]

    def __iter__(self):
        return iter(self.interactions)

    def save(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "interaction_history.pkl")
        logger.info(f"save interaction history to {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(self.interactions, f)
