from abc import ABC, abstractmethod

from attr import define

from .pose import Agent2DPose, Agent3DPose


@define
class AgentState:
    pose_2d: Agent2DPose
    pose_3d: Agent3DPose
    camera_height: float = 0.88  # in meter


class AgentEnv(ABC):
    """
    Base agent that can interact with the environment
    """

    @abstractmethod
    def _prepare_env(self, env_config, *args, **kwargs):
        """
        1. set env config.
        2. initialize env that can interact with agent.
        3. get ground truth semantics
        """

    @abstractmethod
    def _prepare_occupancy_map(self, *args, **kwargs):
        """
        Here we use frontier-based exploration policy.
        """

    @abstractmethod
    def _prepare_low_level_planner(self, *args, **kwargs):
        """
        low-level PointNav
        """

    @abstractmethod
    def _prepare_agent_state(self, *args, **kwargs):
        """
        pose, inventory, etc.
        """

    @abstractmethod
    def _observation_wrapper(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    def _prepare_perception(self, *args, **kwargs):
        """grounding SAM"""

    def _prepare_vlmap(self, *args, **kwargs):
        """vlmap is running background"""

    def _prepare_memory(self, *args, **kwargs):
        """
        Memory that agent can store experiences
        """
