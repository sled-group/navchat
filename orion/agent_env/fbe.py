"""
Frontier Based Exploration
This is used to explore the environment, collecting rgbd data and build vlmap.
"""

import numpy as np

from orion import logger
from orion.abstract.pose import Agent2DPose, FrontierWayPoint
from orion.agent_env.habitat.base import HabitatAgentEnv
from orion.agent_env.habitat.utils import try_action_import_v2
from orion.utils import visulization as vis

MOVE_FORWARD, _, TURN_LEFT, TURN_RIGHT, STOP = try_action_import_v2()


class FBEAgentEnv(HabitatAgentEnv):
    def __init__(self, fast_explore: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fast_explore = fast_explore

    def move_and_spin(self, spin_angle=360, move_ahead=0):
        for _ in range(move_ahead):
            self.step(MOVE_FORWARD)
        if spin_angle > 0:
            for _ in range(spin_angle // self.config.SIMULATOR.TURN_ANGLE):
                self.step(TURN_RIGHT)
        else:
            for _ in range(-spin_angle // self.config.SIMULATOR.TURN_ANGLE):
                self.step(TURN_LEFT)

    def loop(self):
        self.fbe.reset(reset_floor=True, init_spin=True)
        while self.step_count < 500:
            if self.fbe.mode == self.fbe.InitSpinMode:
                self.move_and_spin(360)
                self.fbe._init_check_large_room()
                self.fbe.mode = self.fbe.ExploreMode
                self.fbe.set_explore_strategy(self.fast_explore)

            elif self.fbe.mode == self.fbe.ExploreMode:
                self.follower.set_traversible_map(self.fbe.traversable_map)
                navigable_mask = self.follower.get_navigable_mask()
                if not self.fbe.fast_explore:
                    # go to viewpt, look around, then go to goalpt, look around
                    next_plan: FrontierWayPoint = self.fbe.plan(
                        navigable_mask, with_viewpoint=True
                    )
                else:
                    # go to goalpt directly, update at every serveral steps
                    next_plan: FrontierWayPoint = self.fbe.plan(
                        navigable_mask, with_viewpoint=False
                    )
                if next_plan is None:
                    logger.info("all goal finished===")
                    self.fbe.reset(reset_floor=True, init_spin=True)
                    break
                else:
                    if not self.fbe.fast_explore:
                        viewpt: Agent2DPose = next_plan.viewpt
                        if viewpt is not None:
                            logger.info(f"=== move to view point first {viewpt}===")

                            best_action = self._get_next_action(viewpt)
                            if best_action == 0:
                                logger.info(
                                    "cannot change position with the follower==="
                                )
                            while best_action != 0:
                                self.step(best_action)
                                best_action = self._get_next_action(viewpt)
                            self.move_and_spin(-90)
                            self.move_and_spin(180)

                        goalpt: Agent2DPose = next_plan.goalpt
                        if goalpt is not None:
                            logger.info(f"=== move to goal point {goalpt}===")
                            best_action = self._get_next_action(
                                goalpt, new_goal_dist=10
                            )
                            if best_action == 0:
                                logger.info(
                                    "cannot change position with the follower==="
                                )
                            while best_action != 0:
                                self.step(best_action)
                                best_action = self._get_next_action(
                                    goalpt, new_goal_dist=10
                                )
                            self.move_and_spin(-90)
                            self.move_and_spin(180)

                    else:
                        goalpt: Agent2DPose = next_plan.goalpt
                        if goalpt is None:
                            continue
                        logger.info(f"=== move to goal point {goalpt}===")
                        best_action = self._get_next_action(goalpt, new_goal_dist=10)
                        count = self.fbe.fbecfg.fast_explore_forwardcount
                        if best_action == 0:
                            logger.info("cannot change position with the follower===")
                        while best_action != 0 and count > 0:
                            self.step(best_action)
                            if best_action == MOVE_FORWARD:
                                count -= 1
                            best_action = self._get_next_action(
                                goalpt, new_goal_dist=10
                            )

                        if (
                            self.fbe.l2(goalpt, self._grdpose)
                            < self.fbe.fbecfg.dist_large_thres
                        ):
                            self.move_and_spin(-90)
                            self.move_and_spin(180)


if __name__ == "__main__":
    from orion.config.my_config import SCENE_ID_FLOOR_SET

    game = FBEAgentEnv(
        total_round=1,
        scene_ids=["4ok3usBNeis"],
        floor_set=(-1, 1),
        fast_explore=True,
        display_shortside=256,
        save_dir_name="predict",
        auto_record=True,
        record_dir="recordings_prelim_fbe",
        display_setting="rgb+occumap+topdownmap",
        use_gt_pose=False,
        load_existing_occumap=True,
        save_new_occumap=False,
    )
    game.run()
