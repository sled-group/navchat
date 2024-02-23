import copy
import os
import random
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from skimage.morphology import disk
from sklearn.cluster import AgglomerativeClustering

from orion import logger
from orion.abstract.agent import AgentState
from orion.abstract.pose import Agent2DPose, Agent3DPose, FrontierWayPoint, GoalArea
from orion.config.my_config import FBEConfig, MapConfig
from orion.map.occupancy import OccupancyMapping
from orion.navigation.waypoint_planner import PointPlanner
from orion.utils import visulization as vis
from orion.utils.geometry import CoordinateTransform
from orion.utils.geometry import PinholeCameraModel as pinhole


class FrontierBasedExploration:
    InitSpinMode = 0  # usually for the beginning
    ExploreMode = 1  # fbe explore
    ExploitMode = 2  # usually pointNav directly

    fast_explore = False

    def __init__(
        self,
        mapcfg: MapConfig,
        init_agtpose: Agent3DPose,
        fbecfg: FBEConfig = FBEConfig(),
    ):
        self.mapcfg = mapcfg
        self.fbecfg = fbecfg
        num_grid = mapcfg.num_grid
        cell_size = mapcfg.cell_size
        self.cam_insc = pinhole.get_camera_intrinsic_matrix(
            mapcfg.screen_h, mapcfg.screen_w, mapcfg.fov
        )
        self.cam_insc_inv = np.linalg.inv(self.cam_insc)

        self.occu_maping = OccupancyMapping(mapcfg)

        self.init_agtpose = init_agtpose
        self.transform_fn = CoordinateTransform(num_grid, cell_size, init_agtpose)
        self.init_grdpose = self.transform_fn.agt2grd_pose(init_agtpose)

        self.frontier_means: List[Tuple[Union[int, np.int32], bool]] = []

        self.reset("", False)

        self.reference_map = None

    def set_explore_strategy(self, fast_explore: bool):
        self.fast_explore = fast_explore

    def occu_map(self):
        return self.occu_maping.map

    def reset(
        self,
        occupancy_map_path: str = "",
        reset_floor: bool = False,
        init_spin: bool = True,
        set_reference: bool = False,
    ):
        if os.path.exists(occupancy_map_path):
            self.occu_maping.load(occupancy_map_path)
            logger.info("load existing occumap from {} ...".format(occupancy_map_path))
            set_reference = True

        if set_reference:
            self.set_reference_map()

        if reset_floor:
            self.occu_maping.reset_floor()

        if init_spin:
            self.mode = FrontierBasedExploration.InitSpinMode

        self.grd_view: Optional[Agent2DPose] = None
        self.grd_goal: Optional[Agent2DPose] = None

        self.plan_history: List[Tuple[int, int]] = []
        self.agtpose_history: List[Agent2DPose] = []
        self.go_to_center = None

    def set_reference_map(self):
        self.reference_map = self.occu_maping.map.copy()

    def find_frontiers(self) -> List[Tuple[Union[int, np.int32], bool]]:
        yy, xx = np.where(self.occu_maping.map == OccupancyMapping.FRONTIER)

        candidates = np.stack([xx, yy], axis=1)
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None, linkage="single", distance_threshold=1.5
            ).fit(candidates)
        except:
            return []

        frontiers = {}  # type: ignore

        for i, c in enumerate(clustering.labels_):
            if c in frontiers:
                frontiers[c].append(candidates[i])
            else:
                frontiers[c] = [candidates[i]]
        frontier_means = []

        for k in frontiers:
            points = np.vstack(frontiers[k])
            mean = np.mean(points, axis=0)
            squared_distances = np.sum((points - mean) ** 2, axis=1)
            variance = np.sqrt(np.mean(squared_distances))

            if (
                len(frontiers[k]) >= self.fbecfg.size_thres
                and variance >= self.fbecfg.var_thres
            ) or len(frontiers[k]) >= self.fbecfg.size_large_thres:
                # if hitwall, then the frontier point is deprioritized
                _, reached = PointPlanner.line_search(
                    self.current_grdpose[0],
                    self.current_grdpose[1],
                    mean[0],
                    mean[1],
                    self.wall_mask,
                    stop_at_wall=True,
                )
                frontier_means.append((mean.astype(np.int32), reached))

        if self.fast_explore and len(self.plan_history) > 0:
            last_pt = self.plan_history[-1]
            sorted_frontier_means = sorted(
                frontier_means,
                key=lambda x: self.l2(x[0], self.current_grdpose)  # close to agent
                - x[1] * 10  #  not hit wall
                + self.l2(x[0], last_pt) * 10,  # close to last goal
            )
        else:
            sorted_frontier_means = sorted(
                frontier_means,
                key=lambda x: self.l2(x[0], self.current_grdpose)  # close to agent
                - x[1] * 50,  #  not hit wall
            )

        return [i[0] for i in sorted_frontier_means]

    def update(
        self,
        depth: np.ndarray,
        agent_state: AgentState,
        laser_scan=None,
        is_rotate=False,
        delta_y=0.0,
    ):
        agtpose = agent_state.pose_3d
        self.cam_exsc_inv = self.transform_fn.get_relative_campose(agtpose)
        self.occu_maping.update(
            depth,
            self.cam_exsc_inv,
            self.cam_insc_inv,
            is_rotate=is_rotate,
            camera_height_change=delta_y,
        )
        self.current_grdpose = agent_state.pose_2d
        self.agtpose_history.append(self.current_grdpose)

        # complete the holes or eye blind area in the map
        # use ray shooting for current fov, use breshenham to fill the holes
        grd_x, grd_z, theta = self.current_grdpose.to_list()
        blind_area_grds = PointPlanner.range_search(
            grd_x,
            grd_z,
            theta,
            self.obstacle_mask,
            fov=self.mapcfg.fov,
            dist=self.mapcfg.blind_dist,
        )
        if len(blind_area_grds) > 0:
            self.occu_maping.map[
                blind_area_grds[:, 1], blind_area_grds[:, 0]
            ] = OccupancyMapping.FREE

        if laser_scan is not None:
            laser_navi_area, laser_scope_mask = laser_scan
            z, x = np.where(laser_navi_area)
            self.occu_maping.map[z, x] = OccupancyMapping.FREE
            non_navi_mask = np.logical_and(~laser_navi_area, laser_scope_mask)
            false_floor_mask = np.logical_and(
                non_navi_mask, self.working_navigable_mask
            )
            z, x = np.where(false_floor_mask)
            self.occu_maping.map[z, x] = OccupancyMapping.OCCUPIED

            # wheel radius should be free space
            wheel_radius = int(self.mapcfg.wheel_radius // self.mapcfg.cell_size) + 1
            selem = disk(wheel_radius, dtype=np.bool8)
            z, x = np.where(selem)
            self.occu_maping.map[
                z + grd_z - wheel_radius, x + grd_x - wheel_radius
            ] = OccupancyMapping.FREE

        self.frontier_means = self.find_frontiers()

    def get_goal_area(self, img_mask: np.ndarray) -> Optional[GoalArea]:
        """given the masked pixels, return the goal area in the grid map"""
        img_mask = img_mask.reshape(-1)
        cam_pts = self.occu_maping.cam_pts[:, img_mask]
        wld_pts = pinhole.cam2wld(cam_pts, cam_pose=self.cam_exsc_inv)
        grd_xs = np.round(
            wld_pts[0, :] / self.mapcfg.cell_size + self.mapcfg.num_grid // 2
        ).astype(np.int32)
        grd_zs = np.round(
            self.mapcfg.num_grid // 2 - wld_pts[2, :] / self.mapcfg.cell_size
        ).astype(np.int32)

        height = -np.mean(wld_pts[1, :]) + self.mapcfg.camera_height
        # get the contour and the center
        contour_mask = np.zeros_like(self.occu_maping.map, dtype=np.uint8)
        contour_mask[grd_zs, grd_xs] = 1
        contour_mask = cv2.dilate(
            contour_mask, np.ones((1, 1), dtype=np.uint8), iterations=2
        )
        contours, _ = cv2.findContours(
            contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) == 0:
            logger.info(" [fbe goal area] no contours found")
            return None
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour = sorted_contours[0]
        M = cv2.moments(contour)
        if M["m00"] == 0:
            logger.info(" [fbe goal area] no contours found")
            return None
        cen_x = int(M["m10"] / M["m00"])
        cen_z = int(M["m01"] / M["m00"])

        return GoalArea(x=cen_x, z=cen_z, contour=contour, height=height)

    @property
    def working_navigable_mask(self):
        return np.logical_or(
            self.occu_maping.map == OccupancyMapping.FREE,
            self.occu_maping.map == OccupancyMapping.FRONTIER,
        )

    @property
    def traversable_map(self):
        # to set transversable map for fmm planner
        if self.reference_map is None:
            return self.working_navigable_mask
        else:
            reachable_area = (self.occu_maping.map == OccupancyMapping.UNKNOWN) & (
                self.reference_map == OccupancyMapping.FREE
            )
            return np.logical_or(self.working_navigable_mask, reachable_area)

    def _init_check_large_room(self):
        # just finish
        occumap_free_mask = self.occu_maping.map == OccupancyMapping.FREE
        occumap_free_mask = occumap_free_mask.astype(np.uint8)
        contours, _ = cv2.findContours(
            occumap_free_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if len(contours) > 0:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contour = sorted_contours[0]
            M = cv2.moments(contour)
            if M["m00"] > 1200:
                cen_x = int(M["m10"] / M["m00"])
                cen_z = int(M["m01"] / M["m00"])
                self.go_to_center = FrontierWayPoint(
                    viewpt=None,
                    goalpt=Agent2DPose(x=cen_x, z=cen_z),
                    fast_explore=self.fast_explore,
                )
                logger.info(
                    f" [fbe init] large room {M['m00']}, go to center {cen_x, cen_z}"
                )

    def plan(
        self, navigable_mask=None, with_viewpoint=True
    ) -> Optional[FrontierWayPoint]:
        # navigable_mask is given by fmm planner
        # first plan view point, then target point
        if self.go_to_center is not None:
            waypt = copy.deepcopy(self.go_to_center)
            self.go_to_center = None
            return waypt

        # if navigable_mask is not None:
        #     navigable_mask = navigable_mask & self.traversable_map
        # else:
        #     navigable_mask = self.traversable_map

        navigable_mask = vis.get_largest_connected_area(navigable_mask, erosion_iter=2)
        navigable_mask = navigable_mask.astype(np.bool8)

        while len(self.frontier_means) > 0:
            logger.info(f" [fbe plan] {len(self.frontier_means)} frontiers left")
            next_frontier = self.frontier_means.pop(0)
            cen_x, cen_z = next_frontier
            logger.info(f" [fbe plan] next frontier {cen_x, cen_z}")

            wall_mask = (self.occu_maping.map == OccupancyMapping.WALL) | (
                self.occu_maping.map == OccupancyMapping.UNKNOWN
            )

            if with_viewpoint:
                if (
                    self.l2((cen_x, cen_z), self.current_grdpose)
                    < self.fbecfg.dist_small_thres
                ):  # avoid loop in one frontier
                    logger.info(
                        f" [fbe slow explore plan] too close to the frontier, skip  {self.l2((cen_x, cen_z), self.current_grdpose)} [1]",
                    )
                    continue

                if any(
                    [
                        self.l2((cen_x, cen_z), h) < self.fbecfg.dist_large_thres
                        for h in self.plan_history
                    ]
                ):
                    logger.info(
                        " [fbe slow explore plan] too close to the previous goal, skip [2]"
                    )
                    continue

                self.grd_view = PointPlanner.plan_view_point(
                    cen_x, cen_z, navigable_mask, wall_mask, radius_1=25, radius_2=35
                )
                if self.grd_view is not None:
                    if any(
                        [
                            self.l2(self.grd_view, h) < self.fbecfg.dist_small_thres
                            for h in self.plan_history
                        ]
                    ):
                        logger.info(
                            " [fbe slow explore plan] too close to the previous goal, skip [3]"
                        )
                        self.grd_view = None
                    else:
                        self.plan_history.append((self.grd_view.x, self.grd_view.z))

                self.grd_goal = PointPlanner.plan_view_point(
                    cen_x, cen_z, navigable_mask, wall_mask, radius_1=1, radius_2=20
                )

                # self.grd_goal = PointPlanner.plan_reachable_point(
                #     cen_x, cen_z, navigable_mask, max_radius=20
                # )
                if self.grd_goal is None:  # no reachable point
                    logger.info(" [fbe slow explore plan] no reachable point, skip [4]")
                    continue

                if any(
                    [
                        self.l2(self.grd_goal, h) < self.fbecfg.dist_small_thres
                        for h in self.plan_history
                    ]
                ):
                    logger.info(
                        " [fbe slow explore plan] too close to the previous goal, skip [1]"
                    )
                    continue
                if self.l2((cen_x, cen_z), self.grd_goal) < 10:
                    self.plan_history.append(
                        (int(cen_x), int(cen_z))
                    )  # also add to avoid loop
                self.plan_history.append((self.grd_goal.x, self.grd_goal.z))

            else:
                self.grd_goal = PointPlanner.plan_view_point(
                    cen_x, cen_z, navigable_mask, wall_mask, radius_1=1, radius_2=20
                )
                # navigable_mask_img = np.zeros(shape=(*navigable_mask.shape, 3), dtype=np.uint8)
                # navigable_mask_img[navigable_mask] = (255, 255, 255)
                # navigable_mask_img[wall_mask] = (0, 255, 0)
                # cv2.circle(navigable_mask_img, (cen_x, cen_z), 1, (0, 0, 255), thickness=2)
                # vis.plot_uint8img_with_plt(navigable_mask_img, title="navigable_mask_img")

                if self.grd_goal is None:  # no reachable point
                    logger.info(" [fbe fast explore plan] no reachable point, skip [1]")
                    continue

                if (
                    len(self.plan_history) > 2
                    and self.l2(self.grd_goal, self.plan_history[-2]) < 5
                    and self.l2(self.plan_history[-2], self.plan_history[-1]) > 8
                ):
                    logger.info(" [fbe fast explore plan] potiential loop, skip [2]")
                    continue

                if len(self.plan_history) > 2:
                    _, is_in_sight = PointPlanner.line_search(
                        self.grd_goal.x,
                        self.grd_goal.z,
                        self.current_grdpose.x,
                        self.current_grdpose.z,
                        wall_mask,
                        stop_at_wall=True,
                    )
                    if (
                        is_in_sight
                        and self.l2(self.current_grdpose, self.grd_goal) < 20
                        and self.l2(self.plan_history[-1], self.grd_goal) < 3
                        and self.l2(self.plan_history[-2], self.grd_goal) < 3
                    ):
                        logger.info(
                            " [fbe fast explore plan] potiential loop, skip [3]"
                        )
                        continue

                    if len(self.agtpose_history) >= 20 and len(self.plan_history) >= 3:
                        pose = self.agtpose_history[-1]
                        if (
                            all(
                                [
                                    self.l2(pose, h) < 8
                                    for h in self.agtpose_history[-20:]
                                ]
                            )
                            and all(
                                [
                                    self.l2(self.grd_goal, h) < 8
                                    for h in self.plan_history[-3:]
                                ]
                            )
                            and self.l2(pose, self.grd_goal) < 25
                        ):
                            logger.info(
                                " [fbe fast explore plan] potiential loop, skip [4]"
                            )
                            self.plan_history.append((self.grd_goal.x, self.grd_goal.z))
                            continue

                if len(self.agtpose_history) > 1 and any(
                    [self.l2(self.grd_goal, h) < 10 for h in self.agtpose_history]
                ):
                    logger.info(" [fbe fast explore plan] potiential loop, skip [5]")
                    continue

                self.plan_history.append((self.grd_goal.x, self.grd_goal.z))

            return FrontierWayPoint(
                viewpt=self.grd_view,
                goalpt=self.grd_goal,
                fast_explore=self.fast_explore,
            )

        # fast explore in exploitation mode may slip some area, only when using gtpose
        if self.reference_map is not None:
            # compare the origin map and the current map
            # if a large area is not explored, then go to the center of the area
            occumap_free_mask = self.occu_maping.map == OccupancyMapping.FREE
            origin_free_mask = self.reference_map == OccupancyMapping.FREE
            diff_mask = origin_free_mask & np.logical_not(occumap_free_mask)
            # find the largest connected area
            diff_mask = diff_mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if len(contours) > 0:
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                contour = sorted_contours[0]
                M = cv2.moments(contour)
                if M["m00"] > 200:
                    cen_x = int(M["m10"] / M["m00"])
                    cen_z = int(M["m01"] / M["m00"])
                    canvas = np.zeros_like(diff_mask, dtype=np.uint8)
                    # add some fake frontiers, so that the robot will go to check
                    cv2.drawContours(
                        canvas, [contour], -1, 1, thickness=cv2.FILLED
                    )  # free space
                    cv2.drawContours(
                        canvas, [contour], -1, 2, thickness=3
                    )  # frontier space

                    # random split the fontier lines
                    zmax = contour[:, 0, 1].max()
                    zmin = contour[:, 0, 1].min()
                    xmax = contour[:, 0, 0].max()
                    xmin = contour[:, 0, 0].min()
                    for _ in range(min(int(M["m00"] // 100), 5)):
                        x = random.randint(xmin, xmax)
                        z = random.randint(zmin, zmax)
                        frotier_mask = canvas == 2
                        split_mask = np.zeros_like(frotier_mask, dtype=np.bool)
                        split_mask[z - 5 : z + 5, :] = 1
                        split_mask[:, x - 5 : x + 5] = 1
                        canvas[np.logical_and(frotier_mask, split_mask)] = 1

                    # update occu_map
                    self.occu_maping.map[canvas == 1] = OccupancyMapping.FREE
                    self.occu_maping.map[canvas == 2] = OccupancyMapping.FRONTIER
                    self.frontier_means = self.find_frontiers()

                    navigable_mask = vis.get_largest_connected_area(
                        self.occu_maping.map == OccupancyMapping.FREE, erosion_iter=2
                    )
                    self.grd_goal = PointPlanner.plan_reachable_point(
                        cen_x, cen_z, navigable_mask, max_radius=50
                    )

                    logger.info(
                        f" [fbe fast explore plan] complementary {self.grd_goal}, area diff {M['m00']}"
                    )
                    return FrontierWayPoint(
                        viewpt=None,
                        goalpt=self.grd_goal,
                        fast_explore=self.fast_explore,
                    )

        return None

    @property
    def map_bound(self):
        z_indices, x_indices = np.where(self.occu_maping.map > 0)
        xmin = np.min(x_indices)
        xmax = np.max(x_indices)
        zmin = np.min(z_indices)
        zmax = np.max(z_indices)
        return xmin, xmax, zmin, zmax

    def display_occumap(self, potential_goal_area=None):
        xmin, xmax, zmin, zmax = self.map_bound

        crop_map = self.occu_maping.map[zmin : zmax + 1, xmin : xmax + 1]

        grd_x, grd_z, theta = self.current_grdpose.to_list()

        grd_x, grd_z = grd_x - xmin, grd_z - zmin

        # make color
        color_map = np.zeros((crop_map.shape[0], crop_map.shape[1], 3), dtype=np.uint8)
        for i in range(5):
            color_map[crop_map == i] = OccupancyMapping.color(i)

        if potential_goal_area is not None:
            for goal_area in potential_goal_area:
                contour = copy.deepcopy(goal_area.contour)
                contour[:, 0, 0] -= xmin
                contour[:, 0, 1] -= zmin
                cv2.drawContours(color_map, [contour], 0, (0, 255, 255), -1)

        # draw agent
        cv2.circle(color_map, (int(grd_x), int(grd_z)), 4, (122, 122, 0), -1)
        agent_arrow = vis.get_contour_points((grd_x, grd_z, theta), size=6)
        cv2.drawContours(color_map, [agent_arrow], 0, (0, 122, 122), -1)

        # draw frontier center
        for v in self.frontier_means:
            cv2.circle(
                color_map, (int(v[0] - xmin), int(v[1] - zmin)), 5, (255, 255, 0), -1
            )

        # draw goal
        if self.grd_goal:
            cv2.circle(
                color_map,
                (int(self.grd_goal.x - xmin), int(self.grd_goal.z - zmin)),
                5,
                (255, 0, 255),
                -1,
            )

        # draw view point
        if self.grd_view:
            cv2.circle(
                color_map,
                (int(self.grd_view.x - xmin), int(self.grd_view.z - zmin)),
                5,
                (0, 255, 255),
                -1,
            )

        return color_map

    @property
    def topdown_map_base(self):
        if self.reference_map is not None:
            return self.reference_map.copy()
        else:
            return self.occu_maping.map.copy()

    def _reset_to_existing_map(self):
        self.occu_maping.map = self.reference_map.copy()

    @staticmethod
    def l2(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    @property
    def wall_mask(self):
        return self.occu_maping.map == OccupancyMapping.WALL

    @property
    def obstacle_mask(self):
        return np.logical_or(
            self.occu_maping.map == OccupancyMapping.OCCUPIED, self.wall_mask
        )
