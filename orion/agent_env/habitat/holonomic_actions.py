import cv2
import habitat
import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat.tasks.nav.nav import SimulatorTaskAction


@habitat.registry.register_action_space_configuration(name="Holonomic")
class HolonomicMovement(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()
        config[HabitatSimActions.MOVE_BACKWARD] = habitat_sim.ActionSpec(
            "move_backward",
            habitat_sim.ActuationSpec(amount=self.config.FORWARD_STEP_SIZE),
        )
        return config


@habitat.registry.register_task_action
class MoveBackwardAction(SimulatorTaskAction):
    name = "MOVE_BACKWARD"

    def _get_uuid(self, *args, **kwargs) -> str:
        return "move_backward"

    def step(self, *args, **kwargs):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        return self._sim.step(HabitatSimActions.MOVE_BACKWARD)


HabitatSimActions.extend_action_space("MOVE_BACKWARD")
