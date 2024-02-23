import copy
import math

import matplotlib.pyplot as plt
import numpy as np
import skfmm
from numpy import ma

from orion import logger
from orion.abstract.pose import Agent2DPose

try:
    from habitat.sims.habitat_simulator.actions import HabitatSimActions

    MOVE_FORWARD = HabitatSimActions.MOVE_FORWARD
    TURN_LEFT = HabitatSimActions.TURN_LEFT
    TURN_RIGHT = HabitatSimActions.TURN_RIGHT
    STOP = HabitatSimActions.STOP
except:
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    STOP = 3

ACTION_DICT = {
    "MOVE_FORWARD": MOVE_FORWARD,
    "TURN_LEFT": TURN_LEFT,
    "TURN_RIGHT": TURN_RIGHT,
    "STOP": STOP,
}

INV_ACTION_DICT = {v: k for k, v in ACTION_DICT.items()}
NEAR_REWARD = 1.314
COLLISION_PENALTY = 1000


class FMMPlanner:
    def __init__(
        self,
        num_rots=360 // 15,
        step_size=0.25 / 0.05,
        goal_radius=0.3 / 0.05,
        wheel_radius=0.2 / 0.05,
    ):
        # TODO: need a collision recovery mechanism
        # the occupancy map is not accurate due to the habitat simulator
        """
        Args:
            traversible: 2D numpy binary array of traversible space. (TODO remember contains agent size)
            num_rots: Number of rotations to complete a circle.
            step_size: step size in grid units.
            goal_radius: radius of goal in grid units.
        """
        self.step_size = int(step_size)
        self.num_rots = int(num_rots)
        self.goal_radius = math.ceil(goal_radius)
        self.wheel_radius = math.ceil(wheel_radius)

        self.action_list = self.prepare_cadidate_actions()

    def set_traversible_map(self, traversible):
        self.original_traversible = traversible.copy()
        # # # erode the traversible to aviod potential collision
        # for _ in range(self.wheel_radius//2):
        #     traversible_map = skimage.morphology.binary_erosion(
        #         traversible_map, skimage.morphology.disk(1)
        #     )

        # # get largest connected component contour, like cv2.findContours
        # traversible_map = traversible_map.astype(np.uint8)
        # contours, _ = cv2.findContours(
        #     traversible_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        # )
        # traversible_ = np.zeros_like(traversible_map)
        # contour = sorted(contours, key=cv2.contourArea)[-1]
        # cv2.drawContours(traversible_, [contour], -1, 1, -1)

        self.traversible_map = traversible.copy()

    def set_goal(self, goal: Agent2DPose):
        """set the goal and compute the distance map"""
        traversible_ma = ma.masked_values(self.traversible_map * 1, 0)
        traversible_ma[goal.z, goal.x] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        # dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd

    def prepare_cadidate_actions(self):
        action_list = [[MOVE_FORWARD]]  # [[MOVE_FORWARD], [STOP]]
        append_list_pos = []
        append_list_neg = []
        for _ in range(self.num_rots):
            append_list_pos.append(TURN_LEFT)
            append_list_neg.append(TURN_RIGHT)
            action_list.append(append_list_pos[:] + [MOVE_FORWARD])
            action_list.append(append_list_neg[:] + [MOVE_FORWARD])
        return action_list

    def _virtual_steps(self, a_list, state: Agent2DPose, pre_collision_dict=None):
        traversible = self.traversible_map
        goal_dist = self.fmm_dist
        boundary_limits = np.array(traversible.shape)[::-1]
        x, z, t = state.x, state.z, state.t
        if t is None:
            t = 0.0
        out_states = []
        cost_start = goal_dist[int(z), int(x)]  # Actual distance in cm.
        collision_reward = 0
        for i in range(len(a_list)):
            action = a_list[i]
            x_new, y_new, t_new = x * 1.0, z * 1.0, t * 1.0
            if action == MOVE_FORWARD:
                angl = t
                x_new = x + np.cos(angl) * self.step_size
                y_new = z + np.sin(angl) * self.step_size
                t_new = angl
            elif action == TURN_LEFT:
                t_new = t - 2.0 * np.pi / self.num_rots
            elif action == TURN_RIGHT:
                t_new = t + 2.0 * np.pi / self.num_rots

            collision_reward = -1
            inside = np.all(
                np.array([int(x_new), int(y_new)]) < np.array(boundary_limits)
            )
            inside = inside and np.all(
                np.array([int(x_new), int(y_new)]) >= np.array([0, 0])
            )
            _new_state = Agent2DPose(x, z, t)

            if inside:
                not_collided = True
                if action == MOVE_FORWARD:
                    for s in np.linspace(0, 1, self.step_size + 2):
                        _x = x * s + (1 - s) * x_new
                        _z = z * s + (1 - s) * y_new
                        not_collided = not_collided and traversible[int(_z), int(_x)]
                        if not_collided is False:
                            break
                if not_collided:
                    collision_reward = 0
                    x, z, t = x_new, y_new, t_new  # type: ignore
                    _new_state = Agent2DPose(x, z, t)
            out_states.append(_new_state)

        cost_end = goal_dist[int(z), int(x)]
        reward_near_goal = 0.0
        if cost_end < self.step_size:
            reward_near_goal = NEAR_REWARD
        costs = cost_end - cost_start

        if pre_collision_dict is not None:
            angle, collision = self.match_collision_dict(pre_collision_dict, a_list)
            if collision:
                collision_reward = costs - COLLISION_PENALTY
            else:
                collision_reward = 0

        reward = -costs + reward_near_goal + collision_reward
        return reward, (out_states)

    def find_best_action_set(self, state: Agent2DPose, pre_collision_dict=None):
        best_a_list = [MOVE_FORWARD]
        best_reward, state_list = self._virtual_steps(
            best_a_list, state, pre_collision_dict
        )
        if pre_collision_dict is None:
            best_reward = best_reward + 0.1

        st_lsts, rews = [], []
        for a_list in self.action_list:
            rew, st_lst = self._virtual_steps(a_list, state, pre_collision_dict)
            rews.append(rew)
            # Prefer shorter action sequences.
            rew = rew - len(st_lst) * 0.1
            st_lsts.append(st_lst)
            if rew > best_reward:
                best_a_list = a_list
                best_reward = rew
                state_list = st_lst

        # logger.info(f"[FMMplanner] best reward {best_reward}")
        if all(
            [r == NEAR_REWARD or r == 0 or r < -COLLISION_PENALTY // 2 for r in rews]
        ):
            # unreachable or already near goal
            return [STOP], [state]
        return best_a_list, state_list

    def match_collision_dict(self, pre_collision_dict, action_list):
        if action_list == [STOP]:
            return 0, False
        if action_list == [MOVE_FORWARD]:
            return 0, pre_collision_dict[0]
        angle = 0
        for a in action_list:
            if a == TURN_LEFT:
                angle -= 360 // self.num_rots
            elif a == TURN_RIGHT:
                angle += 360 // self.num_rots
        # wrap angle in [-180, 180]
        angle = (angle + 180) % 360 - 180
        return angle, pre_collision_dict[angle]

    def is_navigable(self, state: Agent2DPose):
        """determine if the state is naviable"""
        if state is None:
            return False
        if self.original_traversible[int(state.z), int(state.x)]:
            return True
        return False

    def is_near_goal(self, state: Agent2DPose, goal: Agent2DPose, goal_dist=None):
        """determine if the state is close to the goal"""
        x, y = state.x, state.z
        gx, gy = goal.x, goal.z
        dist = np.sqrt((x - gx) ** 2 + (y - gy) ** 2)
        if goal_dist is not None:
            if dist < goal_dist:
                return True
            return False
        else:
            if dist < self.goal_radius:
                return True
            return False

    def get_action(
        self,
        start: Agent2DPose,
        goal: Agent2DPose,
        set_goal=True,
        pre_collision_dict=None,
        goal_dist=None,
    ):
        if self.is_near_goal(start, goal, goal_dist=goal_dist):
            logger.info("[FMMplanner] Already near goal, stop")
            return STOP, start
        if not self.is_navigable(start) or not self.is_navigable(goal):
            logger.info(
                f"[FMMplanner] Not navigable. start {start} {self.is_navigable(start)}, goal {goal} {self.is_navigable(goal)}"
            )
            return STOP, start
        if set_goal:
            self.set_goal(goal)
        _ = self.find_best_action_set(start, pre_collision_dict=pre_collision_dict)
        next_act, next_state = _[0][0], _[1][0]
        return next_act, next_state

    def plan(self, start: Agent2DPose, goal: Agent2DPose, plot=True):
        state = copy.deepcopy(start)
        states = []
        a_list = []
        reachable = False
        for i in range(1000):
            if i == 0:
                a, state = self.get_action(state, goal, True)
            else:
                a, state = self.get_action(state, goal, False)
            states.append(state)
            a_list.append(a)
            if a == 0:
                reachable = self.is_near_goal(state, goal)
                break
        if plot:
            self.plot(start, goal, states)
        return reachable, states, a_list

    def plot(self, start, goal, states):
        im = self.original_traversible.copy().astype(np.uint8)
        new_y, new_x = np.where(self.traversible_map)
        im[new_y, new_x] = 2

        z_indices, x_indices = np.where(im > 0)
        xmin = np.min(x_indices)
        xmax = np.max(x_indices)
        zmin = np.min(z_indices)
        zmax = np.max(z_indices)
        im = im[zmin : zmax + 1, xmin : xmax + 1]
        fig, ax = plt.subplots()
        ax.imshow(im * 1.0, vmin=-0.5, vmax=1.5)
        ax.plot(goal.x - xmin, goal.z - zmin, "bx")
        ax.plot(start.x - xmin, start.z - zmin, "rs")
        states = [(s.x - xmin, s.z - zmin) for s in states]
        states = np.array(states)
        ax.plot(states[:, 0], states[:, 1], "r.")
        plt.show()
