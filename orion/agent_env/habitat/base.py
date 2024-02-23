import os
import shutil
from copy import deepcopy
from typing import Optional

import cv2
import habitat
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import append_text_to_image, draw_collision
from habitat_sim.utils.common import d3_40_colors_rgb
from scipy.spatial.transform import Rotation as R

from orion import logger
from orion.abstract.agent import AgentEnv, AgentState
from orion.abstract.interfaces import Observations
from orion.abstract.pose import Agent2DPose, Agent3DPose
from orion.config.my_config import FBEConfig, MapConfig
from orion.map.occupancy import OccupancyMapping
from orion.navigation.frontier_based_exploration import FrontierBasedExploration
from orion.navigation.shortest_path_follower_wrapper import HabitatFollower, MyFollower
from orion.utils import visulization
from orion.utils.file_load import get_floor_set_str
from orion.utils.geometry import CoordinateTransform

from . import holonomic_actions, utils


class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


class HabitatAgentEnv(AgentEnv):
    def __init__(
        self,
        config_path="orion/config/my_objectnav_hm3d.yaml",
        map_cfg: MapConfig = MapConfig(),
        split="val",
        scene_ids=["4ok3usBNeis"],
        is_objectnav=True,
        display_shortside=480,
        save_dir_name="sandbox",  # prefix of the save dir
        auto_record=False,  # # whether to record the episode
        record_dir="recordings",  # the directory to save the recordings
        display_setting="rgb+semantic",
        display_horizontally=True,
        headless=False,
        floor_set=(-1, 1),
        use_gt_pose=False,
        load_existing_occumap=False,
        save_new_occumap=False,
    ) -> None:
        np.set_printoptions(precision=3, suppress=True, linewidth=200)

        self.display_shortside = display_shortside
        self.display_setting = display_setting.split("+")
        self.display_horizontally = display_horizontally
        self.headless = headless
        self.auto_record = auto_record  # This recording is for building vlmap. Not need if vlmap is built
        self.use_gt_pose = use_gt_pose
        self.load_existing_occumap = load_existing_occumap
        self.save_new_occumap = (
            save_new_occumap  # This is only need for fbe. Not need if occumap is built
        )

        logger.info(f"[Init] scene_ids: {scene_ids[0]}, floor_set: {floor_set}")
        self.scene_id_basic = scene_ids[0]

        self._prepare_env(
            config_path,
            is_objectnav=is_objectnav,
            split=split,
            scene_ids=scene_ids,
            map_cfg=map_cfg,
            floor_set=floor_set,
        )

        self.save_dir = "data/experiments/{}_{}_{}".format(
            save_dir_name, self.scene_id, get_floor_set_str(floor_set)
        )
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if auto_record:
            self.recordings = {  # type: ignore
                "depth": [],
                "rgb": [],
                "semantic": [],
                "pose": [],
                "screenshot": [],
            }
            self.record_save_dir = os.path.join(self.save_dir, record_dir)
            if os.path.exists(self.record_save_dir):
                u = input(
                    f"record dir  {self.record_save_dir} already exists, do you want to overwrite? (y/n)"
                )
                if u == "n":
                    assert False, f"record dir already exists {self.record_save_dir}"
                else:
                    shutil.rmtree(self.record_save_dir)
                    os.makedirs(self.record_save_dir)
            else:
                os.makedirs(self.record_save_dir)

        self.next_contour_to_show = None  # cv2.contour
        self.next_pt_to_show: Optional[Agent2DPose] = None  # (x, z)
        self.next_bbox_to_show = None
        self.add_text = None

        self.will_collide = False
        self.collide_tolerance = 5
        self.collide_count = 0

    def reset(self):
        obs = self._set_floor_plan()
        self.step_count = 0
        self._prepare_agent_state()
        self.observations = self._observation_wrapper(obs, None)
        self._prepare_low_level_planner()
        self._prepare_occupancy_map()
        self._prepare_vlmap()
        self._prepare_memory()
        self.display(wait_time=1000)
        self._check_pre_collision()

    def step(self, action):
        """interact with env, update agent state and display"""
        if action == HabitatSimActions.STOP:
            self.env.step(action)
            return
        collide, obs, info = self._step_wrapper(action)
        if collide:
            logger.info("collision with move forward! skip this action")
            if self.pre_collision_dict is not None:
                self.pre_collision_dict[0] = True
            return

        self.step_count += 1
        self._update_agent_state(action, info)
        self.observations = self._observation_wrapper(obs, info)
        self._update_occupancy_map(
            is_rotate=action
            in [HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT]
        )
        self.display()
        self._check_pre_collision(action)

    def _step_wrapper(self, action):
        if self.use_gt_pose:
            obs, _, _, info = self.env.step(action)
            return False, obs, info
        _habitat_state = self.env.habitat_env.sim.get_agent_state()
        obs, _, _, info = self.env.step(action)
        _next_habitat_state = self.env.habitat_env.sim.get_agent_state()
        # Big change of camera height will cause occumap error.
        # We have a floor tolerance of 0.3m. But habitat have navigable area has sudden height change
        # We use the simulator height to migitate this problem.
        # In real robot, this is not an issue, because Locobot can not reach such height.
        self.delta_y = _next_habitat_state.position[1] - self.init_sim_state.position[1]
        if (
            info is not None
            and "collisions" in info
            and info["collisions"]["is_collision"]
        ):
            # if still has collision, reset the agent state to the last state
            self.env.habitat_env.sim.set_agent_state(
                position=_habitat_state.position, rotation=_habitat_state.rotation
            )
            self.will_collide = True
            return True, obs, info
        self.will_collide = False
        return False, obs, info

    def _check_pre_collision(self, action=None):
        if self.use_gt_pose:
            return
        # Collision will cause unexpected error for occupancy map building due to the inaccurate pose estimation
        # Since habitat scan env has many mesh stitch bugs. navigatable area is not consistent with RGBD input
        # so we perform a pre-collision check.
        # Note in real robot, this can be done by using lidar sensor or any laser scan module.

        # rotation check, simplicy rotate the pre_collision_dict from last turn
        if (
            action in [HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT]
            and self.pre_collision_dict is not None
        ):
            last_pre_collision_dict = self.pre_collision_dict.copy()
            pre_collision_dict = {}
            if action == HabitatSimActions.TURN_LEFT:
                for k, v in last_pre_collision_dict.items():
                    angle = k + self.turn_angle
                    angle = (angle + 180) % 360 - 180
                    pre_collision_dict[angle] = v
            else:
                for k, v in last_pre_collision_dict.items():
                    angle = k - self.turn_angle
                    angle = (angle + 180) % 360 - 180
                    pre_collision_dict[angle] = v
            self.pre_collision_dict = pre_collision_dict
            return

        # For each direction (360//turn_angle)
        # check whether move forward will collide
        pre_collision_dict = {}

        _habitat_state = deepcopy(self.env.habitat_env.sim.get_agent_state())
        agt_state = Agent3DPose.from_habitat(_habitat_state)
        agt_T = agt_state.get_T()

        # turn right
        rot_deg = -self.config.SIMULATOR.TURN_ANGLE
        rot_mat = R.from_euler("y", rot_deg, degrees=True).as_matrix()
        rot_T = np.eye(4)
        rot_T[:3, :3] = rot_mat

        cum_rot_T = np.eye(4)
        for i in range(0, 360, self.turn_angle):
            if i > 0:
                cum_rot_T = cum_rot_T @ rot_T

            flag = False
            for s in np.linspace(0, self.config.SIMULATOR.FORWARD_STEP_SIZE, 7):
                move_T = np.eye(4)
                move_T[2, 3] -= s
                _T = agt_T @ cum_rot_T @ move_T
                sim_pos = _T[:3, 3]
                if not self.env.habitat_env.sim.is_navigable(sim_pos):
                    flag = True
            i = (i + 180) % 360 - 180
            if i in [-180, 180]:
                pre_collision_dict[180] = flag
                pre_collision_dict[-180] = flag
            else:
                pre_collision_dict[i] = flag

        self.pre_collision_dict = pre_collision_dict

    def run(self):
        self.reset()
        self.loop()
        self.save()
        self.close()

    def close(self):
        self.env.close()

    def loop(self):
        """logic of how to make actions"""
        return NotImplementedError

    def save(self):
        if self.auto_record:
            self.save_recordings()

        if self.save_new_occumap:
            path = os.path.join(self.save_dir, "occupancy_map.npy")
            logger.info(f"save new occupancy map ... to {path}")
            if self.fbe.reference_map is not None:
                occu_save = self.fbe.reference_map.copy()
            else:
                occu_save = self.fbe.occu_maping.map.copy()
            with open(path, "wb") as f:
                np.save(f, occu_save)

            topdownmap = self.draw_topdown_map(occu_save)
            path = os.path.join(self.save_dir, "occupancy_map.png")
            logger.info(f"save new occupancy map ... to {path}")
            cv2.imwrite(path, utils.transform_rgb_bgr(topdownmap))

    def _prepare_agent_state(self):
        if self.use_gt_pose:
            self._agtpose = Agent3DPose.from_habitat(
                self.env.habitat_env.sim.get_agent_state()
            )
        else:
            self._agtpose = Agent3DPose(position=[0, 0, 0], rotation=[0, 0, 0, 1])
            self.T = np.eye(4)
        self.transform_fn = CoordinateTransform(
            self.mapcfg.num_grid, self.mapcfg.cell_size, self._agtpose
        )
        self._grdpose = self.transform_fn.agt2grd_pose(self._agtpose)
        self.agent_state = AgentState(
            pose_2d=self._grdpose,
            pose_3d=self._agtpose,
            camera_height=self.config.SIMULATOR.AGENT_0.HEIGHT,
        )
        self._print_agent_state()

    def _prepare_low_level_planner(self):
        self._gt_navigate_mask = self._get_gt_navigable_mask(
            os.path.join(self.save_dir, "gt_navigable_mask.npy")
        )  # used for collision check
        self.delta_y = 0.0  # camera height change
        if self.use_gt_pose:
            self.follower = HabitatFollower(
                env_sim=self.env.habitat_env.sim,
                navigatable_mask=self._gt_navigate_mask,
                transform_fn=self.transform_fn,
            )
        else:
            self.follower = MyFollower(
                num_rots=360 // self.config.SIMULATOR.TURN_ANGLE,
                step_size=self.config.SIMULATOR.FORWARD_STEP_SIZE
                / self.mapcfg.cell_size,
                goal_radius=0.5 / self.mapcfg.cell_size,
                wheel_radius=self.config.SIMULATOR.AGENT_0.RADIUS
                / self.mapcfg.cell_size,
            )
        self.pre_collision_dict = None

    def _get_gt_navigable_mask(self, navigatable_mask_path):
        if os.path.exists(navigatable_mask_path):
            navigable_mask = np.load(navigatable_mask_path)
            logger.info(
                f"load navigatable mask from {navigatable_mask_path}, shape {navigable_mask.shape}"
            )
            navigable_mask = visulization.get_largest_connected_area(navigable_mask)

        else:
            num_grid = self.mapcfg.num_grid
            navigable_mask = np.zeros((num_grid, num_grid), dtype=np.bool)
            _init_state = Agent3DPose.from_habitat(
                self.env.habitat_env.sim.get_agent_state()
            )
            _transform_fn = CoordinateTransform(
                self.mapcfg.num_grid, self.mapcfg.cell_size, _init_state
            )

            for grd_z in range(num_grid):
                for grd_x in range(num_grid):
                    sim_pos = _transform_fn.grd2agt_pos(Agent2DPose(grd_x, grd_z))
                    if self.env.habitat_env.sim.is_navigable(sim_pos):
                        navigable_mask[grd_z, grd_x] = 1

            with open(navigatable_mask_path, "wb") as f:
                np.save(f, navigable_mask)
                logger.info(
                    "save navigatable mask to {} ...".format(navigatable_mask_path)
                )
                dir_path = os.path.dirname(navigatable_mask_path)
                visulization.plot_uint8img_with_plt(
                    navigable_mask,
                    os.path.join(dir_path, "navigatable_mask"),
                    save=True,
                )
        return navigable_mask

    def _prepare_occupancy_map(self):
        self.fbe = FrontierBasedExploration(
            mapcfg=self.mapcfg,
            fbecfg=FBEConfig(),
            init_agtpose=self._agtpose,
        )
        if self.load_existing_occumap:
            self.fbe.reset(os.path.join(self.save_dir, "occupancy_map.npy"))

        self._update_occupancy_map()

    def _update_occupancy_map(self, is_rotate=False):
        if not self.use_gt_pose:
            # To aviod potentioal collision, we will perform a pre-collision check for each direction
            # Note in real robot, this can be done by the laser scan module.
            # Here we use the neighbor gt navigatable area as a proxy for simplicity
            cen_x = self.agent_state.pose_2d.x
            cen_z = self.agent_state.pose_2d.z
            radius = self.mapcfg.laser_scan_limit

            laser_navi_area = np.zeros(
                shape=(self.mapcfg.num_grid, self.mapcfg.num_grid), dtype=np.bool
            )
            laser_scope_mask = np.zeros(
                shape=(self.mapcfg.num_grid, self.mapcfg.num_grid), dtype=np.bool
            )

            L = np.arange(-radius, radius + 1)
            X, Z = np.meshgrid(L, L)

            for loc_x, loc_z in zip(X.flatten(), Z.flatten()):
                glb_x, glb_z = int(cen_x + loc_x), int(cen_z + loc_z)
                if (
                    0 <= glb_x < self.mapcfg.num_grid
                    and 0 <= glb_z < self.mapcfg.num_grid
                    and loc_x**2 + loc_z**2 <= radius**2
                ):
                    laser_scope_mask[glb_z, glb_x] = 1
                    if self._gt_navigate_mask[glb_z, glb_x] == 1:
                        laser_navi_area[glb_z, glb_x] = 1

            self.fbe.update(
                self.observations.depth,
                self.agent_state,
                (laser_navi_area, laser_scope_mask),
                is_rotate,
                self.delta_y,
            )
        else:
            self.fbe.update(self.observations.depth, self.agent_state)
        self.follower.set_traversible_map(self.fbe.traversable_map)

    def _prepare_env(self, config_path, *args, **kwargs):
        utils.quiet()

        config = habitat.get_config(config_path)
        utils.update_holonomic_action(config)
        if kwargs["is_objectnav"]:
            utils.update_scene(config, kwargs["split"], kwargs["scene_ids"])
        utils.add_top_down_map_and_collision(config)
        self.env = SimpleRLEnv(config=config)
        self.config = config

        self.mapcfg: MapConfig = kwargs["map_cfg"]
        self.mapcfg.update_with_habitat_config(self.config)

        self.floor_set = kwargs["floor_set"]

        self.max_depth = self.config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH

        if kwargs["is_objectnav"]:
            self.scene_id = self.env.current_episode.scene_id.split("/")[-1].replace(
                ".basis.glb", ""
            )
            logger.info(f"current scene: {self.scene_id}")
            self.obj2cls_dict = self._load_semantic_annoations(kwargs["split"])

        self.turn_angle = self.config.SIMULATOR.TURN_ANGLE
        self.agent_height = self.config.SIMULATOR.AGENT_0.HEIGHT
        self.step_size = self.config.SIMULATOR.FORWARD_STEP_SIZE
        self.FOV = self.config.SIMULATOR.RGB_SENSOR.HFOV

    def display(self, wait_time=1):
        outputs = []
        observations: Observations = self.observations

        rgb_img = utils.transform_rgb_bgr(observations.rgb)
        info = observations.info
        if info and "collisions" in info and info["collisions"]["is_collision"]:
            draw_collision(rgb_img)
        outputs.append(rgb_img)

        if "depth" in self.display_setting:
            depth_img = cv2.applyColorMap(
                (observations.depth / self.max_depth * 255).astype(np.uint8),
                cv2.COLORMAP_JET,
            )
            outputs.append(depth_img)
        if "semantic" in self.display_setting:
            semantic_img = self._draw_segmentation_with_labels(observations.semantic)
            outputs.append(semantic_img)

        if "occumap" in self.display_setting:  # for debug
            occumap = self.fbe.display_occumap()
            if self.display_horizontally:
                resiz_occumap = self.resize_image(
                    utils.transform_rgb_bgr(occumap), height=rgb_img.shape[0]
                )
            else:
                resiz_occumap = self.resize_image(
                    utils.transform_rgb_bgr(occumap), width=rgb_img.shape[1]
                )
            outputs.append(resiz_occumap)

        if "topdownmap" in self.display_setting:  # for demo
            topdownbase = self.fbe.topdown_map_base
            topdownmap = self.draw_topdown_map(topdownbase)
            if self.display_horizontally:
                resiz_occumap = self.resize_image(
                    utils.transform_rgb_bgr(topdownmap), height=rgb_img.shape[0]
                )
            else:
                resiz_occumap = self.resize_image(
                    utils.transform_rgb_bgr(topdownmap), width=rgb_img.shape[1]
                )
            outputs.append(resiz_occumap)

        if self.display_horizontally:
            display_image = np.concatenate(outputs, axis=1)
        else:
            display_image = np.concatenate(outputs, axis=0)

        if self.next_bbox_to_show is not None:
            for bbox, phrase, _ in self.next_bbox_to_show:
                cv2.rectangle(
                    display_image,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 255),
                    thickness=2,
                )
                cv2.putText(
                    display_image,
                    phrase,
                    (bbox[0], bbox[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    3,
                )
        if self.display_horizontally:
            display_image = self.resize_image(
                display_image, height=self.display_shortside
            )
        else:
            display_image = self.resize_image(
                display_image, width=self.display_shortside
            )
        if self.add_text is not None:
            display_image = append_text_to_image(display_image, self.add_text)

        if not self.headless:
            cv2.imshow("display", display_image)
            cv2.waitKey(wait_time)
        self.display_image = display_image

        if self.auto_record:
            self.record()

    def draw_topdown_map(self, topdownbase):
        z_indices, x_indices = np.where(topdownbase > 0)
        xmin = np.min(x_indices)
        xmax = np.max(x_indices)
        zmin = np.min(z_indices)
        zmax = np.max(z_indices)

        # if xmax-xmin < 200:
        #     xmin = max(0, xmin-100)
        #     xmax = min(self.mapcfg.num_grid-1, xmax+100)

        # if zmax-zmin < 200:
        #     zmin = max(0, zmin-100)
        #     zmax = min(self.mapcfg.num_grid-1, zmax+100)

        crop_map = topdownbase[zmin : zmax + 1, xmin : xmax + 1]
        # logger.info(f"crop topdown map to ({xmin}, {zmin}) - ({xmax}, {zmax}))")

        grdpose = self.agent_state.pose_2d
        grd_x, grd_z, theta = grdpose.x, grdpose.z, grdpose.t

        grd_x, grd_z = grd_x - xmin, grd_z - zmin

        # make color
        color_map = np.zeros((crop_map.shape[0], crop_map.shape[1], 3), dtype=np.uint8)
        color_map[crop_map == OccupancyMapping.UNKNOWN] = (130, 130, 130)
        color_map[crop_map == OccupancyMapping.FREE] = (255, 255, 255)
        color_map[crop_map == OccupancyMapping.UNKNOWN_FREE] = (200, 200, 200)
        color_map[crop_map == OccupancyMapping.OCCUPIED] = (150, 220, 150)
        color_map[crop_map == OccupancyMapping.WALL] = (0, 0, 0)
        color_map[crop_map == OccupancyMapping.FRONTIER] = (255, 255, 255)

        if self.next_contour_to_show is not None:
            if (
                isinstance(self.next_contour_to_show, np.ndarray)
                and len(self.next_contour_to_show.shape) == 3
            ):
                contour_show = self.next_contour_to_show - np.array([xmin, zmin])
                cv2.drawContours(color_map, [contour_show], 0, (200, 0, 200), -1)
            elif isinstance(self.next_contour_to_show, list):  # list of contours
                for contour in self.next_contour_to_show:
                    contour_show = contour - np.array([xmin, zmin])
                    cv2.drawContours(color_map, [contour_show], 0, (200, 0, 200), -1)

        # _navigable_mask_crop = self._navigable_mask[zmin:zmax+1, xmin:xmax+1]
        # color_map[np.logical_and(_navigable_mask_crop == 1, crop_map == OccupancyMapping.FREE)] = (200, 200, 200)

        # draw next pt
        if self.next_pt_to_show:
            cv2.circle(
                color_map,
                (
                    int(self.next_pt_to_show.x - xmin),
                    int(self.next_pt_to_show.z - zmin),
                ),
                5,
                (0, 255, 255),
                -1,
            )

        # draw agent
        agent_arrow = visulization.get_contour_points((grd_x, grd_z, theta), size=6)
        cv2.drawContours(color_map, [agent_arrow], 0, (0, 0, 255), -1)

        return color_map

    def _update_agent_state(self, action=None, info=None):
        """return relative campose array"""
        if self.use_gt_pose:
            _sim_agent_state = self.env.habitat_env.sim.get_agent_state()
            self._agtpose = Agent3DPose.from_habitat(_sim_agent_state)
        else:
            if info and "collisions" in info and info["collisions"]["is_collision"]:
                pass
            elif action == HabitatSimActions.MOVE_FORWARD:
                mat = np.eye(4)
                mat[2, 3] -= self.config.SIMULATOR.FORWARD_STEP_SIZE
                self.T = self.T @ mat
            elif action == "MOVE_BACKWARD":
                mat = np.eye(4)
                mat[2, 3] += self.config.SIMULATOR.FORWARD_STEP_SIZE
                self.T = self.T @ mat
            elif action == HabitatSimActions.TURN_LEFT:
                rot_deg = self.config.SIMULATOR.TURN_ANGLE
                rot_mat = R.from_euler("y", rot_deg, degrees=True).as_matrix()
                mat = np.eye(4)
                mat[:3, :3] = rot_mat
                self.T = self.T @ mat
            elif action == HabitatSimActions.TURN_RIGHT:
                rot_deg = -self.config.SIMULATOR.TURN_ANGLE
                rot_mat = R.from_euler("y", rot_deg, degrees=True).as_matrix()
                mat = np.eye(4)
                mat[:3, :3] = rot_mat
                self.T = self.T @ mat
            else:
                pass
            self._agtpose = Agent3DPose.from_numpy(self.T[:3, 3], self.T[:3, :3])

        self._grdpose = self.transform_fn.agt2grd_pose(self._agtpose)

        self.agent_state.pose_2d = self._grdpose
        self.agent_state.pose_3d = self._agtpose

        self._print_agent_state()

    def _print_agent_state(self):
        logger.info(
            f"Step: {self.step_count}, {str(self.agent_state.pose_3d)}, {self.agent_state.pose_2d}"
        )

    def _set_floor_plan(self):
        obs = self.env.reset()
        logger.info(f"current ep: {self.env.habitat_env._current_episode.episode_id}")
        while not (
            self.floor_set[0]
            < self.env.habitat_env._current_episode.start_position[1]
            < self.floor_set[1]
        ):
            obs = self.env.reset()
            logger.info(
                f"current ep: {self.env.habitat_env._current_episode.episode_id}"
            )
        self.init_sim_state = self.env.habitat_env.sim.get_agent_state()
        return obs

    def _observation_wrapper(self, obs, info) -> Observations:
        _obs = deepcopy(obs)
        _info = deepcopy(info)
        rel_cam_pose = self.transform_fn.get_relative_campose(self._agtpose)
        observations = Observations(
            rgb=_obs["rgb"],
            depth=_obs["depth"] * float(self.max_depth),  # habitat depth is normalized
            semantic=_obs["semantic"],
            rel_cam_pose=rel_cam_pose,
            info=_info,
            compass=_obs["compass"],
            gps=_obs["gps"],
        )
        return observations

    def _load_semantic_annoations(self, split):
        scene_data_path = f"data/scene_datasets/hm3d_v0.2/{split}"
        for d in os.listdir(scene_data_path):
            if os.path.isdir(os.path.join(scene_data_path, d)):
                for file in os.listdir(os.path.join(scene_data_path, d)):
                    if file == self.scene_id + ".semantic.txt":
                        semantic_path = os.path.join(scene_data_path, d, file)
                        break

        label_dic = {}
        default_labels = [
            "misc",
            "wall",
            "floor",
            "chair",
            "bed",
            "door",
            "table",
            "couch",
            "toilet",
            "ceiling",
        ]
        for l in default_labels:
            label_dic[l] = len(label_dic)

        obj2cls_dic = {}
        with open(semantic_path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line == "HM3D Semantic Annotations":
                    continue
                object_id, _, label, _ = line.split(",")
                label = label.strip('"')
                if label in label_dic:
                    obj2cls_dic[int(object_id)] = (label_dic[label], label)
                else:
                    if label == "unknown":
                        obj2cls_dic[int(object_id)] = (label_dic["misc"], "misc")
                    elif len(label.split()) >= 3:
                        obj2cls_dic[int(object_id)] = (label_dic["misc"], "misc")
                    else:
                        label_dic[label] = len(label_dic)
                        obj2cls_dic[int(object_id)] = (label_dic[label], label)

        return obj2cls_dic

    @staticmethod
    def resize_image(img, height=None, width=None):
        if height is not None:
            old_h, old_w, _ = img.shape
            if old_h == height:
                return img
            width = int(float(height) / old_h * old_w)
            # cv2 resize (dsize is width first)
            new_img = cv2.resize(
                img,
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )
            return new_img
        elif width is not None:
            old_h, old_w, _ = img.shape
            if old_w == width:
                return img
            height = int(float(width) / old_w * old_h)
            # cv2 resize (dsize is width first)
            new_img = cv2.resize(
                img,
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )
            return new_img

    def _draw_segmentation_with_labels(self, semantic_image):
        semantic_mat = semantic_image.squeeze()
        colored_image = d3_40_colors_rgb[semantic_mat % 40]  # simple 40 color palette
        for obj_id in np.unique(semantic_mat):
            if obj_id == 0:
                continue
            mask = semantic_mat == obj_id
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) > 0:
                contour = sorted(contours, key=cv2.contourArea)[-1]
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                label = self.obj2cls_dict[obj_id][1]
                if label == "misc":
                    continue
                cv2.putText(
                    colored_image,
                    label,
                    (centroid_x, centroid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                )

        return colored_image

    def record(self):
        self.recordings["depth"].append(self.observations.depth.squeeze())
        self.recordings["rgb"].append(self.observations.rgb)
        self.recordings["semantic"].append(self.observations.semantic.squeeze())
        self.recordings["pose"].append(self._agtpose.to_str())
        self.recordings["screenshot"].append(
            utils.transform_rgb_bgr(self.display_image)
        )

    def save_recordings(self):
        # This is for building vlmap
        assert (
            len(self.recordings["depth"])
            == len(self.recordings["rgb"])
            == len(self.recordings["semantic"])
            == len(self.recordings["pose"])
        )
        logger.info(
            f"save {len(self.recordings['depth'])} frames to {self.record_save_dir}"
        )
        for key, value in self.recordings.items():
            os.makedirs(os.path.join(self.record_save_dir, key), exist_ok=True)
            if key == "pose":
                for i, pose in enumerate(value):
                    with open(
                        os.path.join(
                            self.record_save_dir, key, f"{self.scene_id}_{i}.txt"
                        ),
                        "w",
                    ) as f:
                        f.write(pose)
            elif key == "depth":
                for i, data in enumerate(value):
                    with open(
                        os.path.join(
                            self.record_save_dir, key, f"{self.scene_id}_{i}.npy"
                        ),
                        "wb",
                    ) as f:
                        # np.save(f, data)
                        np.save(f, (data * 1000).astype(np.uint16))  # save storage
            elif key == "semantic":
                for i, data in enumerate(value):
                    with open(
                        os.path.join(
                            self.record_save_dir, key, f"{self.scene_id}_{i}.npy"
                        ),
                        "wb",
                    ) as f:
                        np.save(f, data.astype(np.uint16))
            else:  # save as image
                for i, img in enumerate(value):
                    cv2.imwrite(
                        os.path.join(
                            self.record_save_dir, key, f"{self.scene_id}_{i}.png"
                        ),
                        utils.transform_rgb_bgr(img),
                    )

        # save semantic labels
        with open(os.path.join(self.record_save_dir, "obj2cls_dict.txt"), "w") as f:
            f.write("0: 0, misc\n")  # habitat semantic label 0 is background
            for key, value in self.obj2cls_dict.items():
                f.write(f"{key}: {value[0]}, {value[1]}\n")

    @property
    def navigable_mask(self):
        return self.follower.get_navigable_mask()

    def _get_next_action(self, goalpt: Agent2DPose, new_goal_dist=None):
        """
        return the best action to reach the goal
        new_goal_dist: the distance to the goalpt, if None, use the default goal_radius in follower
        """
        goalpt = self.follower.revise_pose(goalpt)
        self.next_pt_to_show = goalpt  # show point in the topdown map
        best_action = self.follower.get_next_action(
            start=self._grdpose,
            goal=goalpt,
            pre_collision_dict=self.pre_collision_dict,
            goal_dist=new_goal_dist,
        )
        return best_action
