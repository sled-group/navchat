import os

import numpy as np

from orion.abstract.pose import Agent2DPose
from orion.navigation.fmm_planner import INV_ACTION_DICT, FMMPlanner

data_dir = "data/experiments/predict_4ok3usBNeis_B1_U1"

occumap_mask = np.load(os.path.join(data_dir, "occupancy_map.npy"))
im = occumap_mask == 1  # floor

planner = FMMPlanner()
planner.set_traversible_map(im)

y, x = np.where(planner.traversible_map)

while True:
    goal_ind = np.random.choice(y.size)
    start_ind = np.random.choice(y.size)

    goal = Agent2DPose(x[goal_ind], y[goal_ind], 0)
    start = Agent2DPose(x[start_ind], y[start_ind], -np.pi / 2)

    print(f"start: {start}, goal: {goal}")
    reachable, states, a_list = planner.plan(start, goal, plot=True)
    # red square is the start
    # blue cross is the goal
    # red line is the planned path
    print(reachable, [INV_ACTION_DICT[a] for a in a_list])
