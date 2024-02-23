import random
from typing import Optional

import cv2
import numpy as np

from orion import logger
from orion.abstract.pose import Agent2DPose


class PointPlanner:
    """Grid-based point planner"""

    @staticmethod
    def circle_mask(array, center, radius):
        y, x = np.indices(array.shape)
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        mask = distance <= radius
        return mask

    @staticmethod
    def ring_mask(array, center, radius_1, radius_2):
        y, x = np.indices(array.shape)
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        mask = np.logical_and(distance <= radius_2, distance >= radius_1)
        return mask

    @staticmethod
    def line_search(
        x0, y0, x1, y1, wall_mask=None, stop_at_wall=False, object_contours=None
    ):
        # whether intersect with wall (or other occupancy objets) using breshenham_search
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        count = 0
        line = []

        if object_contours is not None:  # sometimes object are mis-detected by walls
            if isinstance(object_contours, np.ndarray):
                object_contours = [object_contours]
            _wall_mask = np.zeros_like(wall_mask, dtype=np.uint8)
            cv2.drawContours(_wall_mask, object_contours, -1, 1, -1)
            # expant a little bit
            _wall_mask = cv2.dilate(_wall_mask, np.ones((1, 1), np.uint8), iterations=1)
            wall_mask = np.logical_and(
                wall_mask, np.logical_not(_wall_mask.astype(np.bool))
            )

        while x0 != x1 or y0 != y1:
            count += 1
            if wall_mask is None or wall_mask[y0, x0] == 0:
                line.append((x0, y0))
            if stop_at_wall and wall_mask is not None and wall_mask[y0, x0] == 1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        if stop_at_wall:
            reached = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) < 4
            return line, reached
        else:
            ratio = len(line) / count
            return line, ratio > 0.95

    @staticmethod
    def range_search(cen_x, cen_z, theta, obstacle_mask, fov=90, dist=25):
        # get the brim of the pie
        start_theta = theta - np.deg2rad(fov // 2)
        end_theta = theta + np.deg2rad(fov // 2)

        start_pt = np.array(
            [cen_x + np.cos(start_theta) * dist, cen_z + np.sin(start_theta) * dist]
        ).astype(np.int)
        end_pt = np.array(
            [cen_x + np.cos(end_theta) * dist, cen_z + np.sin(end_theta) * dist]
        ).astype(np.int)
        # get all points in  the line using cv2
        tgt_line, _ = PointPlanner.line_search(
            start_pt[0], start_pt[1], end_pt[0], end_pt[1]
        )

        # steps = int(0.5*np.pi*dist)
        # thetas = np.linspace(start_theta, end_theta, steps)

        # xs = np.round(cen_x + np.cos(thetas) * dist).astype(np.int)
        # zs = np.round(cen_z + np.sin(thetas) * dist).astype(np.int)

        complementary_grd = np.empty((0, 2), dtype=np.int)

        for tgt_x, tgt_z in tgt_line:
            line, _ = PointPlanner.line_search(
                cen_x, cen_z, tgt_x, tgt_z, obstacle_mask, stop_at_wall=True
            )
            if line:
                complementary_grd = np.concatenate(
                    (complementary_grd, np.array(line)), axis=0
                )

        # # line = np.array(line) # (N, 2)
        # img = np.ones((obstacle_mask.shape[0], obstacle_mask.shape[1], 3), dtype=np.uint8) *255
        # img[obstacle_mask == 1] = (0, 0, 0)
        # # img[line[:, 1], line[:, 0]] = (0, 255, 0)
        # img[complementary_grd[:, 1], complementary_grd[:, 0]] = (0, 255, 0)

        # cv2.imwrite("testestest.png", img)
        # input ()

        return complementary_grd

    @staticmethod
    def plan_view_point(
        cen_x,
        cen_z,
        navigable_mask,
        wall_mask,
        radius_1=20,
        radius_2=35,
        init_x=None,
        init_z=None,
        strict=True,
    ) -> Optional[Agent2DPose]:
        """given target (cen_x, cen_z), find a view point in the navigatble mask that can see the target point, not blocked by wall"""

        candidate_mask = PointPlanner.ring_mask(
            navigable_mask, (cen_x, cen_z), radius_1=radius_1, radius_2=radius_2
        )
        candidate_mask = candidate_mask & navigable_mask

        if init_x is not None and init_z is not None:
            radius = 4
            init_mask = PointPlanner.circle_mask(
                navigable_mask, (init_x, init_z), radius=radius
            )
            init_mask = init_mask & candidate_mask

            while np.sum(init_mask) == 0 and radius < 20:
                radius += 2
                init_mask = PointPlanner.circle_mask(
                    navigable_mask, (init_x, init_z), radius=radius
                )
                init_mask = init_mask & candidate_mask

            if np.sum(init_mask) > 0:
                candidate_mask = init_mask
            else:
                logger.info(
                    " [pointplanner] no suitable viewpoint arond init point has been found"
                )
                return None

        # img = np.zeros((navigable_mask.shape[0], navigable_mask.shape[1], 3), dtype=np.uint8)
        # img[navigable_mask == 1] = (255, 255, 255)
        # img[wall_mask == 1] = (0, 255, 0)
        # img[candidate_mask] = (0, 0, 255)
        # cv2.imwrite("testestest.png", img)

        if np.sum(candidate_mask) == 0:
            # logger.info(" [pointplanner] no suitable view point")
            return None
        else:
            # try multiple times
            zs, xs = np.where(candidate_mask == 1)
            # logger.info("find {} possible view points".format(len(zs)))
            idxs = list(range(len(zs)))
            random.shuffle(idxs)
            for rand_idx in idxs:
                # rand_idx = random.randint(0, len(ys)-1)
                rand_z, rand_x = zs[rand_idx], xs[rand_idx]
                _, is_in_sight = PointPlanner.line_search(
                    cen_x, cen_z, rand_x, rand_z, wall_mask, stop_at_wall=strict
                )
                if is_in_sight:
                    # return Agent2DPose(x=rand_x, z=rand_z, t=None)
                    return Agent2DPose.from_two_pts([rand_x, rand_z], [cen_x, cen_z])
            # logger.info(" [pointplanner] no suitable view point")
            return None

    @staticmethod
    def plan_reachable_point(
        cen_x,
        cen_z,
        navigable_mask,
        wall_mask=None,
        init_x=None,
        init_z=None,
        max_radius=30,
    ) -> Optional[Agent2DPose]:
        radius = 2
        candidate_mask = PointPlanner.circle_mask(
            navigable_mask, (cen_x, cen_z), radius
        )
        candidate_mask = candidate_mask & navigable_mask
        while np.sum(candidate_mask) == 0 and radius < max_radius:
            radius += 2
            # logger.info(f"increase radius to {radius}")
            candidate_mask = PointPlanner.circle_mask(
                navigable_mask, (cen_x, cen_z), radius
            )
            candidate_mask = candidate_mask & navigable_mask
            # if radius > 16:
            #     logger.info("the goal may be too far to find a navigable point")
            #     break

        if np.sum(candidate_mask) == 0:
            return None

        zs, xs = np.where(candidate_mask == 1)
        rand_idx = random.randint(0, len(zs) - 1)
        rand_z, rand_x = zs[rand_idx], xs[rand_idx]

        # if init_x is not None and init_z is not None and wall_mask is not None:
        #     _, is_in_sight = PointPlanner.line_search(init_x, init_z, rand_x, rand_z, wall_mask, stop_at_wall=True)
        #     while not is_in_sight:
        #         rand_idx = random.randint(0, len(zs)-1)
        #         rand_z, rand_x = zs[rand_idx], xs[rand_idx]
        #         _, is_in_sight = PointPlanner.line_search(init_x, init_z, rand_x, rand_z, wall_mask, stop_at_wall=True)

        return Agent2DPose(x=rand_x, z=rand_z, t=None)

    @staticmethod
    def plan_view_point_with_contour(
        cen_x, cen_z, navigable_mask, wall_mask, contour=None, threshold=12
    ) -> Optional[Agent2DPose]:
        radius = max(int(np.sqrt(cv2.contourArea(contour)) / 2) + 8, threshold)
        init_radius = radius

        _wall_mask = np.zeros_like(wall_mask, dtype=np.uint8)
        cv2.drawContours(_wall_mask, [contour], -1, 1, -1)
        _wall_mask = cv2.dilate(_wall_mask, np.ones((1, 1), np.uint8), iterations=1)
        _wall_mask = np.logical_and(
            wall_mask, np.logical_not(_wall_mask.astype(np.bool8))
        )

        viewpt = PointPlanner.plan_view_point(
            cen_x=cen_x,
            cen_z=cen_z,
            navigable_mask=navigable_mask,
            wall_mask=_wall_mask,
            radius_1=radius,
            radius_2=radius + 5,
            strict=False,
        )
        while viewpt is None:
            radius += 5
            # logger.info(f"to find view pt, increase radius to {radius}")
            viewpt = PointPlanner.plan_view_point(
                cen_x=cen_x,
                cen_z=cen_z,
                navigable_mask=navigable_mask,
                wall_mask=_wall_mask,
                radius_1=radius,
                radius_2=radius + 5,
                strict=False,
            )
            if radius > max(int(2 * init_radius), 40):
                logger.info(f" [pointplanner] cannot find a suitable viewpoint ...")
                return None
        return viewpt
