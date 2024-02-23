import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

from orion.map.occupancy import OccupancyMapping
from orion.navigation.waypoint_planner import PointPlanner

data_dir = "data/experiments/predict_4ok3usBNeis_B1_U1"


occupancy_map = np.load(os.path.join(data_dir, "occupancy_map.npy"))
navigation_mask = np.load(os.path.join(data_dir, "gt_navigable_mask.npy"))


y, x = np.where(navigation_mask == 1)
ymin, ymax = y.min(), y.max()
xmin, xmax = x.min(), x.max()

occupancy_map_crop = occupancy_map[ymin : ymax + 1, xmin : xmax + 1]
navigation_mask_crop = navigation_mask[ymin : ymax + 1, xmin : xmax + 1]

src = (102, 176)
tgt = (46, 164)
pts, reached = PointPlanner.line_search(
    src[0],
    src[1],
    tgt[0],
    tgt[1],
    occupancy_map_crop == OccupancyMapping.WALL,
    stop_at_wall=True,
)
print("Can reach the tgt pt? ", reached)
print("pts along the line:", pts)
pts = np.array(pts)
navigation_mask_crop_color = np.stack(
    [navigation_mask_crop, navigation_mask_crop, navigation_mask_crop], axis=-1
)
navigation_mask_crop_color = navigation_mask_crop_color.astype(np.uint8)
navigation_mask_crop_color = 122 + navigation_mask_crop_color * 122
navigation_mask_crop_color[occupancy_map_crop == OccupancyMapping.WALL] = [0, 0, 0]

cv2.circle(navigation_mask_crop_color, (src[0], src[1]), 2, (0, 255, 0), -1)  # green
cv2.circle(navigation_mask_crop_color, (tgt[0], tgt[1]), 2, (0, 0, 255), -1)  # blue
cv2.polylines(navigation_mask_crop_color, [pts], False, (255, 0, 0), 1)  # red
plt.imshow(navigation_mask_crop_color)
plt.show()
