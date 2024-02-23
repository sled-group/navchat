"""
HybridSearchAgentEnv: memory + VLmap + FBE + CLIP&GroundSAM detect
Can not talk, No ChatGPT control, only cmd mode.
Refer to chatgpt_control.py for the chatgpt control version.
"""

import os
import re
import time
from typing import List, Optional, Union

import cv2
import numpy as np

from orion import logger
from orion.abstract.interfaces import TextQuery
from orion.abstract.perception import MaskedBBOX
from orion.abstract.pose import (
    Agent2DPose,
    FrontierWayPoint,
    GoalArea,
    ObjectWayPoint,
    get_rotation_angle2d,
)
from orion.agent_env.habitat.base import HabitatAgentEnv
from orion.agent_env.habitat.utils import try_action_import_v2
from orion.config.my_config import *
from orion.map.map_search.search_voxel import VLMapSearch
from orion.map.occupancy import OccupancyMapping
from orion.memory.neural_memory2d import NeuralMemory2D
from orion.navigation.frontier_based_exploration import FrontierBasedExploration
from orion.navigation.waypoint_planner import PointPlanner
from orion.perception.detector.groundingSAM import GroundingSAM
from orion.perception.extractor.clipbase import CLIPBase
from orion.utils import visulization as vis
from orion.utils.clip_score_utils import CLIPScorer

MOVE_FORWARD, _, TURN_LEFT, TURN_RIGHT, STOP = try_action_import_v2()


class HybridSearchBase(HabitatAgentEnv):
    def __init__(
        self,
        use_memory=True,
        use_vlmap=True,
        use_explore=True,
        fast_explore=False,
        load_existing_memory=False,
        is_vlmap_baseline=False,
        is_cow_baseline=False,
        vlmap_dir="lseg_vlmap",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.is_vlmap_baseline = is_vlmap_baseline
        self.is_cow_baseline = is_cow_baseline

        self._prepare_perception()

        self.vlmap_path = os.path.join(self.save_dir, vlmap_dir, "sparse_vxl_map.npz")
        print("VLmap path:", self.vlmap_path)
        self.use_memory = use_memory
        self.use_vlmap = use_vlmap
        self.use_explore = use_explore
        self.fast_explore = fast_explore
        self.load_existing_memory = load_existing_memory

        if self.is_vlmap_baseline:
            self.use_memory = False
            self.use_explore = False
        if self.is_cow_baseline:
            self.use_memory = False
            self.use_vlmap = False
            self.use_explore = True
        if self.is_vlmap_baseline or self.is_cow_baseline:
            self.baseline_mode = True

        logger.info(
            f"[Init] use_memory: {use_memory}, use_vlmap: {use_vlmap}, fast_explore: {fast_explore}, load_existing_memory: {load_existing_memory}"
        )

        self.buff_waypt: List[Union[ObjectWayPoint, FrontierWayPoint]] = []
        self.next_waypt: Union[ObjectWayPoint, FrontierWayPoint] = None

    def _prepare_occupancy_map(self):
        super()._prepare_occupancy_map()
        self.fbe.set_explore_strategy(self.fast_explore)

    def _prepare_perception(self):
        self.detect_results = None

        if not self.is_vlmap_baseline:
            self.clip_detector = CLIPBase(cfg=CLIPConfig_vitL14_datacomp())
            self.clip_scorer = CLIPScorer()
        else:
            self.clip_detector = None
            self.clip_scorer = None

        if not (self.is_cow_baseline or self.is_vlmap_baseline):
            self.groundsam = GroundingSAM()
        else:
            self.groundsam = None

        self.clip_text_list = None
        self.clip_text_embs = None
        self.clip_detect_on = False
        self.sam_detect_on = False
        self.found_goal = False

    def _prepare_vlmap(self):
        self.vlmap_results = None

        if "lseg" in self.vlmap_path:
            self.map_search = VLMapSearch(
                load_sparse_map_path=self.vlmap_path,
            )
        else:
            self.map_search = VLMapSearch(
                clipcfg=CLIPConfig_vitL14_datacomp(),
                load_sparse_map_path=self.vlmap_path,
            )

    def _prepare_memory(self):
        self.posmem_results = None
        self.negmem_results = None

        if not self.use_memory:
            return
        self.object_memory = NeuralMemory2D(
            num_grid=self.mapcfg.num_grid, feat_dim=self.map_search.feat_dim
        )
        self.room_memory = {}  # {room_name:view_point}
        if self.load_existing_memory:
            self.object_memory.load()

    def close(self):
        super().close()
        del self.map_search.clip_model
        del self.clip_detector
        del self.groundsam

    def _memory_retrieve(self, text_query: TextQuery, neg_only=False):
        if text_query is None:
            if neg_only:
                return []
            return [], []
        if not self.use_memory:
            if neg_only:
                return []
            return [], []

        if not neg_only:
            logger.info(f"==== search memory for {text_query} === ")
        else:
            logger.info(f"==== search negative memory for {text_query} === ")
        wall_mask = self.fbe.occu_maping.map == OccupancyMapping.WALL
        navigatable_mask = vis.get_largest_connected_area(self.navigable_mask)

        if text_query.target and text_query.target in text_query.prompt:
            feat_emb = self.map_search.clip_model.encode_text([text_query.prompt])
        else:
            feat_emb = self.map_search.clip_model.encode_text(
                [text_query.prompt, text_query.target]
            )
        feat_emb = feat_emb.mean(axis=0)

        if neg_only:
            negative_mem_waypts = self.object_memory.retrieve_neg(
                feat_emb, navigatable_mask, wall_mask
            )
        else:
            positive_mem_waypts, negative_mem_waypts = self.object_memory.retrieve(
                feat_emb, navigatable_mask, wall_mask
            )

        for pt in negative_mem_waypts:
            pt.type = "mem_neg"

        if neg_only:
            return negative_mem_waypts
        else:
            for pt in positive_mem_waypts:
                pt.type = "mem_pos"
            return positive_mem_waypts, negative_mem_waypts

    def _vlmap_retrieve(self, text_query: TextQuery):
        if not self.use_vlmap:
            return []

        tgt_name = text_query.target

        if tgt_name in ["wall", "floor", "stuff"]:
            return []

        logger.info(f"==== search VLmap for {text_query.prompt} ===")
        xmin, xmax, zmin, zmax = self.fbe.map_bound
        self.map_search.update_index(
            [xmin, xmax, zmin, zmax]
        )  # Has to align with occupancy map

        predict_map, query_labels = self.map_search.query(tgt_name)
        wall_mask = self.fbe.occu_maping.map == OccupancyMapping.WALL
        navigatable_mask = vis.get_largest_connected_area(self.navigable_mask)

        vl_waypts: List[ObjectWayPoint] = self.map_search.plan(
            tgt_name,
            query_labels,
            predict_map,
            navigatable_mask,
            wall_mask,
            show=False,
        )

        for pt in vl_waypts:
            pt.type = "vlmap"

        return vl_waypts

    def _detect_with_clip(self, rgb):
        img_embs = self.clip_detector.encode_image(rgb)
        text_probs = self.clip_detector.score(img_embs, self.clip_text_embs)
        if self.clip_scorer.is_goal_found(text_probs[0][1]):
            return True
        else:
            return False

    def _detect_with_sam(self, rgb, text_query: Optional[TextQuery]):
        if (
            len(self.clip_scorer.prob_list)
            and self.clip_scorer.prob_list[-1] < 0.4
            and not self.clip_scorer.found_goal
        ) or text_query is None:
            return MaskedBBOX(False, [], [], [])
        masked_bbox = self.groundsam.predict(rgb, text_query)
        if bool(masked_bbox) == True:
            logger.info(f"\033[33m [GroundingSAM] detected possible targets\033[m")
            return masked_bbox.de_duplication()
        else:
            logger.info(f"\033[33m [GroundingSAM] does not detect target\033[m")
            return MaskedBBOX(False, [], [], [])

    ### Geometry Utils ###
    def _check_agent_close2contour(self, contour):
        for i in range(len(contour)):
            if (
                np.sqrt(
                    (self._grdpose.x - contour[i][0][0]) ** 2
                    + (self._grdpose.z - contour[i][0][1]) ** 2
                )
                < 15
            ):
                logger.info(" [check agent close to contour] True")
                return True
        logger.info(" [check agent close to contour] False")
        return False

    def _check_2item_close(
        self,
        item1: Union[ObjectWayPoint, GoalArea],
        item2: Union[ObjectWayPoint, GoalArea],
        xz_dist_thres=12,
        h_dist_thres=0.5,
        overlap_thres=0.7,
    ) -> bool:
        if isinstance(item1, GoalArea):
            pt1 = Agent2DPose(x=item1.x, z=item1.z)
            height1 = item1.height
            contour1 = item1.contour
        else:
            pt1 = item1.goalpt
            height1 = item1.height
            contour1 = item1.contour

        if isinstance(item2, GoalArea):
            pt2 = Agent2DPose(x=item2.x, z=item2.z)
            height2 = item2.height
            contour2 = item2.contour
        else:
            pt2 = item2.goalpt
            height2 = item2.height
            contour2 = item2.contour

        xz_dist = self.fbe.l2(pt1, pt2)

        if height1 is not None and height2 is not None:
            h_dist = abs(height1 - height2)
        else:
            h_dist = 0

        overlap = vis.contour_overlap(
            num_grid=self.mapcfg.num_grid, contour1=contour1, contour2=contour2
        )
        is_close = (
            xz_dist < xz_dist_thres and h_dist < h_dist_thres or overlap > overlap_thres
        )

        logger.info(
            f" [check objwpts] overlap: {overlap}, xz_dist: {xz_dist}, h_dist: {h_dist}, {is_close}"
        )
        return is_close

    def _check_agent_close2pose2d(self, pose: Agent2DPose):
        xz_dist = self.fbe.l2(self._grdpose, pose)
        return xz_dist < 3

    ### Deduplicate waypt that is negative ###
    def _remove_negative_waypts(
        self, src_pts: List[ObjectWayPoint], neg_pts: List[ObjectWayPoint]
    ) -> List[ObjectWayPoint]:
        final_output = []
        logger.info(f" remove possible negative waypts ")
        for item in src_pts:
            # too close to the negative memory
            if any(
                [
                    self._check_2item_close(item, item_neg)  # 15, 0.5, 0.67
                    for item_neg in neg_pts
                ]
            ):
                logger.info(f"{item.goalpt} too close to the negative memory, skip")
                continue
            final_output.append(item)

        return final_output


class HybridSearchAgentEnv(HybridSearchBase):
    #### User Interaction ####
    def user_reset_goal(self, utterance):
        # "user issue a new goal"
        if "|" not in utterance:
            text_prompt = utterance.strip()
            tgt_object = utterance.strip()
        else:
            text_prompt = utterance.split("|")[0].strip()
            tgt_object = utterance.split("|")[1].strip()

        text_query: TextQuery = TextQuery(text_prompt, tgt_object)
        self.clip_text_list = ["other", text_query.prompt]
        self.clip_text_embs = self.clip_detector.encode_text(self.clip_text_list)

        self.clip_detect_on = False
        self.sam_detect_on = False
        self.camera_flag = "center"
        self.found_goal = False

        self.clip_scorer.reset()
        self.buff_waypt.clear()
        self.ready_for_nextwaypt = True
        self.next_bbox_to_show: Optional[MaskedBBOX] = None
        self.next_pt_to_show: Optional[Agent2DPose] = None
        self.next_contour_to_show: Optional[np.ndarray] = None

        self.finished = False

        if self.use_memory or self.use_vlmap:
            self.fbe.reset(init_spin=False)
            self.fbe.mode = FrontierBasedExploration.ExploitMode
        else:
            self.fbe.reset(reset_floor=True, init_spin=True)

        self.add_text = "Searching"

        return text_query

    def user_update_memory(
        self, utterance, src_text_query: TextQuery, contour: np.ndarray
    ):
        # contour -> grds
        _grd = np.zeros(
            shape=(self.mapcfg.num_grid, self.mapcfg.num_grid), dtype=np.uint8
        )
        cv2.drawContours(_grd, [contour], -1, 1, -1)
        grds = np.stack(np.where(_grd == 1), axis=1)

        if utterance in ["nothing", "n", ""]:
            print("user delete:", src_text_query.prompt)
            self.object_memory.delete(
                feat_val=self.map_search.clip_model.encode_text(
                    [src_text_query.prompt]
                ),
                grd_zxs=grds,
            )
            return
        if "|" not in utterance:
            text_prompt = utterance.strip()
            tgt_object = utterance.strip()
        else:
            text_prompt = utterance.split("|")[0].strip()
            tgt_object = utterance.split("|")[1].strip()

        text_query: TextQuery = TextQuery(text_prompt, tgt_object)
        print("user update:", src_text_query.prompt, " -> ", text_query.prompt)
        self.object_memory.update(
            neg_feat_val=self.map_search.clip_model.encode_text(
                [src_text_query.prompt]
            ),
            tgt_feat_val=self.map_search.clip_model.encode_text(
                [text_query.prompt, text_query.target]
            ),
            grd_zxs=grds,
        )
        self.posmem_results, self.negmem_results = self._memory_retrieve(src_text_query)
        self.posmem_results = self._remove_negative_waypts(
            self.posmem_results, self.negmem_results
        )

    def user_add_memory(self, text_query: TextQuery, contour: np.ndarray):
        # contour -> grds
        _grd = np.zeros(
            shape=(self.mapcfg.num_grid, self.mapcfg.num_grid), dtype=np.uint8
        )
        cv2.drawContours(_grd, [contour], -1, 1, -1)
        grds = np.stack(np.where(_grd == 1), axis=1)

        print("user add:", text_query.prompt)
        self.object_memory.add(
            feat_val=self.map_search.clip_model.encode_text(
                [text_query.prompt, text_query.target]
            ),
            grd_zxs=grds,
        )

    ### Main Loop ###
    def loop(self):
        # talk with user
        # first search memory, then search VLmap, then use FBE & CLIP to search. After confirm with user, then update the memory
        while True:
            utterance = input(
                "what is the new object you are looking for? (phrase|noun) >>>"
            )
            self.user_text_goal = self.user_reset_goal(utterance)

            self.posmem_results, self.negmem_results = self._memory_retrieve(
                self.user_text_goal
            )
            self.posmem_results = self._remove_negative_waypts(
                self.posmem_results, self.negmem_results
            )

            if self.use_memory:
                logger.info(
                    f"\t Found {len(self.posmem_results)} results in memory ==="
                )
                if len(self.posmem_results) > 0:
                    for item in self.posmem_results:
                        logger.info(f"\t Goal:{item.goalpt}, View:{item.viewpt}")
                    self.buff_waypt.extend(self.posmem_results)
                    self.add_text = "Found in memory"
                    self.display()
                    time.sleep(1)
                else:
                    self.add_text = "Nothing found in memory"
                    self.display()
                    time.sleep(1)

            self.vlmap_results = self._vlmap_retrieve(
                self.user_text_goal
            )  # TODO: search all related objects
            self.vlmap_results = self._remove_negative_waypts(
                self.vlmap_results, self.negmem_results
            )

            if self.use_vlmap:
                logger.info(f"\t Found {len(self.vlmap_results)} results in VLmap ===")
                if len(self.vlmap_results) > 0:
                    for item in self.vlmap_results:
                        logger.info(f"\t Goal:{item.goalpt}, View:{item.viewpt}")
                    self.buff_waypt.extend(self.vlmap_results)
                    self.add_text = "Found in VLmap"
                    self.display()
                    time.sleep(1)
                else:
                    self.add_text = "Nothing found in VLmap"
                    self.display()
                    time.sleep(1)

            self.add_text = "Searching"
            self.miniloop()

            if self.finished:
                self.add_text = "Done"
                self.display()
                continue

            # otherwise start to explore by FBE (default slow explore)
            if self.fbe.mode == FrontierBasedExploration.ExploitMode:
                self.fbe.mode = FrontierBasedExploration.InitSpinMode
                self.miniloop()

    def miniloop(self):
        while (
            len(self.buff_waypt) > 0
            or self.fbe.mode == FrontierBasedExploration.InitSpinMode
        ):
            action_gen = self._low_level_action_generator()
            for action in action_gen:
                if action == STOP:
                    break
                self.step(action=action)

                if (
                    isinstance(self.next_waypt, ObjectWayPoint)
                    and self.next_waypt.type == "vlmap"
                    or isinstance(self.next_waypt, FrontierWayPoint)
                ):
                    self.clip_detect_on = True
                    mbbox = self.egoview_detect()
                    if mbbox is not None:  # detect by ego-view
                        self.detect_results = self.egoview_plan(mbbox)
                        if len(self.detect_results) > 0:
                            logger.info(
                                f"==== found {len(self.detect_results)} results in ego-view ==="
                            )
                            for item in self.detect_results:
                                logger.info(
                                    f"\t Goal:{item.goalpt}, View:{item.viewpt}"
                                )

                            # TODO: here only add the middle one
                            detected_item = self.detect_results[
                                len(self.detect_results) // 2
                            ]
                            self.buff_waypt.insert(0, detected_item)
                            self.add_text = "Double check..."
                            self.next_contour_to_show = detected_item.contour

                            # # if too close to the contour, then not send to buff_waypt
                            # if self._check_close_to_contour(detected_item.contour):
                            #     self.found_goal = True
                            #     # update the contour if this waypt is not sent into buffer
                            #     self.next_contour_to_show = detected_item.contour
                            # else:
                            #     self.found_goal = False

                            # stop the current action_gen
                            # reset next_waypt
                            # goto the next ego-view suggeested waypoint
                            # if not close to the contour\
                            self.ready_for_nextwaypt = True
                            break
                        else:
                            self.clip_scorer.set_masktime(6)
                            self.add_text = "Searching"

                else:
                    # egoview objwpt, memory objwpt do not detect all the time
                    # but still update the clip score
                    self._detect_with_clip(self.observations.rgb)

            if (
                isinstance(self.next_waypt, ObjectWayPoint)
                and self.next_waypt.type == "mem_pos"
            ):
                print("we reach a goal in memory, please give me another goal")
                self.finished = True
                break

            if (
                isinstance(self.next_waypt, ObjectWayPoint)
                and self.next_waypt.type == "ego_view"
            ):
                self.found_goal = True
                # self.next_contour_to_show = detected_item.contour

                # double check with GroundingSAM to see whether the goalarea is still correct
                if self.next_waypt.type == "ego_view":
                    logger.info("\033[33m [GroundingSAM] Double check...\033[m")
                    mbboxes: MaskedBBOX = self._detect_with_sam(
                        self.observations.rgb, self.user_text_goal
                    )
                    self.next_bbox_to_show = mbboxes
                    self.display()
                    self.next_bbox_to_show = None
                    time.sleep(2)

                    tmp_results = self.egoview_plan(mbboxes)
                    if not any(
                        [
                            self._check_2item_close(item, self.next_waypt)
                            for item in tmp_results
                        ]
                    ):
                        self.found_goal = False
                        logger.info(
                            " previous egoview goalarea may not be correct, reschedule current goal area..."
                        )
                        self.add_text = "Searching"

                        # add new detected goalarea to objectwaypts
                        for item in tmp_results:
                            if all(
                                [
                                    not self._check_2item_close(opt, item)
                                    for opt in self.buff_waypt
                                    if isinstance(opt, ObjectWayPoint)
                                ]
                            ):
                                self.buff_waypt.insert(0, item)
                                logger.info(f" add possible new goalarea: {item}")
                    else:
                        detected_item = tmp_results[len(tmp_results) // 2]
                        self.next_contour_to_show = detected_item.contour

            if self.found_goal:
                utterance = input("I found a possible goal, is it correct? (y/n)")
                if utterance == "y":
                    self.user_add_memory(self.user_text_goal, self.next_contour_to_show)
                    self.buff_waypt.clear()
                    self.finished = True
                else:
                    utterance = input(
                        "if not correct, what is this object? (phrase|noun) >>>"
                    )
                    self.user_update_memory(
                        utterance, self.user_text_goal, self.next_contour_to_show
                    )
                    self.clip_scorer.set_masktime(6)  # avoid too frequent detection
                    self.found_goal = False
                    self.add_text = "Searching"

            if self.fbe.mode == FrontierBasedExploration.InitSpinMode:
                self.fbe.mode = FrontierBasedExploration.ExploreMode
                self.fbe._init_check_large_room()

            if self.fbe.mode == FrontierBasedExploration.ExploreMode:
                self.follower.set_traversible_map(self.fbe.traversable_map)
                navigable_mask = self.follower.get_navigable_mask()
                fbe_waypt = self.fbe.plan(
                    navigable_mask, with_viewpoint=not self.fast_explore
                )
                print("fbe_waypt", fbe_waypt)
                if fbe_waypt is not None and not self.finished:
                    if all(
                        [
                            not isinstance(item, FrontierWayPoint)
                            for item in self.buff_waypt
                        ]
                    ):
                        self.buff_waypt.append(fbe_waypt)
                else:
                    logger.info(" ==== FBE finished ====")
                    if not self.finished:
                        logger.info(
                            " sorry I can not find the goal. Please give me another goal."
                        )

    def _low_level_action_generator(self):
        def _spin(angles: List[int]):
            if isinstance(angles, int):
                angles = [angles]
            for angle in angles:
                rot_num = angle // self.turn_angle
                while rot_num != 0:
                    if rot_num > 0:
                        yield TURN_RIGHT
                        rot_num -= 1
                    else:
                        yield TURN_LEFT
                        rot_num += 1

        if self.fbe.mode == FrontierBasedExploration.InitSpinMode:
            self.next_waypt = FrontierWayPoint(
                None, None, fast_explore=self.fast_explore
            )  # placeholder frontier waypt
            self.fbe.reset(reset_floor=True)
            self.fbe.mode = FrontierBasedExploration.ExploreMode
            self.next_pt_to_show = None
            self.next_contour_to_show = None
            self.ready_for_nextwaypt = True
            yield from _spin([360])
            return

        if self.ready_for_nextwaypt:
            self.ready_for_nextwaypt = False
            if len(self.buff_waypt) > 0:
                self.next_waypt = self.buff_waypt.pop(0)
                logger.info(f"\n\n== start to process next waypoint==")
                logger.info(f" {self.next_waypt}")
                logger.info(f" the rest of [{len(self.buff_waypt)}] buff_waypts:")
                for item in self.buff_waypt:
                    logger.info(f" {item}")
                # input("press any key to continue")
            else:
                logger.info("==== all waypoints in the list are finished ===")
                yield STOP

        if isinstance(self.next_waypt, ObjectWayPoint):
            # go to viewpt and look at goalpt
            if self.next_waypt.viewpt is not None:
                self.next_contour_to_show = self.next_waypt.contour
                best_action = self._get_next_action(goalpt=self.next_waypt.viewpt)
                while best_action != STOP:
                    yield best_action
                    best_action = self._get_next_action(goalpt=self.next_waypt.viewpt)
                angle = get_rotation_angle2d(self._grdpose, self.next_waypt.goalpt)
                rot_num = angle // self.turn_angle
                best_action = TURN_RIGHT if rot_num > 0 else TURN_LEFT
                while rot_num != 0:
                    yield best_action
                    angle = get_rotation_angle2d(self._grdpose, self.next_waypt.goalpt)
                    rot_num = angle // self.turn_angle
                    best_action = TURN_RIGHT if rot_num > 0 else TURN_LEFT
            else:
                logger.warning(
                    f"[ObjWayPT] Can not find viewpt for {self.next_waypt.goalpt}!"
                )

        else:
            # fast explore: go to goalpt within at most 8 steps
            if self.next_waypt.fast_explore:
                if self.next_waypt.goalpt is not None:
                    count = self.fbe.fbecfg.fast_explore_forwardcount
                    best_action = self._get_next_action(
                        goalpt=self.next_waypt.goalpt, new_goal_dist=10
                    )
                    while best_action != STOP and count > 0:
                        yield best_action
                        if best_action == MOVE_FORWARD:
                            count -= 1
                        best_action = self._get_next_action(
                            goalpt=self.next_waypt.goalpt, new_goal_dist=10
                        )
                    if (
                        self.fbe.l2(self.next_waypt.goalpt, self._grdpose)
                        < self.fbe.fbecfg.dist_large_thres
                    ):
                        yield from _spin([-90, 180])

            # slow explore: go to viewpt, look around , then go to goalpt, look around
            else:
                if self.next_waypt.viewpt is not None:
                    self.next_pt_to_show = self.next_waypt.viewpt
                    best_action = self._get_next_action(goalpt=self.next_waypt.viewpt)
                    while best_action != STOP:
                        yield best_action
                        best_action = self._get_next_action(
                            goalpt=self.next_waypt.viewpt
                        )
                    yield from _spin([-90, 180])

                if self.next_waypt.goalpt is not None:
                    self.next_pt_to_show = self.next_waypt.goalpt
                    best_action = self._get_next_action(
                        goalpt=self.next_waypt.goalpt, new_goal_dist=10
                    )
                    while best_action != STOP:
                        yield best_action
                        best_action = self._get_next_action(
                            goalpt=self.next_waypt.goalpt, new_goal_dist=10
                        )
                    yield from _spin([-90, 180])

        self.next_pt_to_show = None
        self.next_contour_to_show = None
        self.ready_for_nextwaypt = True
        yield STOP

    ### Ego-view Perception ###
    def egoview_detect(self) -> Optional[MaskedBBOX]:
        # use clip to do first-level detect
        # then use groundSAM to do second-level detect

        if self.clip_detect_on and self._detect_with_clip(self.observations.rgb):
            result: MaskedBBOX = self._detect_with_sam(
                self.observations.rgb, self.user_text_goal
            )
            while bool(result) == True and self._need_adjust_camera(result):
                result = self._detect_with_sam(
                    self.observations.rgb, self.user_text_goal
                )
                self.next_bbox_to_show = result
                self.display()
                self.next_bbox_to_show = None
            if bool(result) == True:
                self.next_bbox_to_show = result
                self.display()
                logger.info(
                    f"\033[33m [GroundingSAM] found bbox in egoview after adjusting camera...\033[m"
                )
                time.sleep(2)
                self.next_bbox_to_show = None
                return result
            else:
                self.clip_scorer.set_masktime(6)  # avoid too frequent detection

        return None

    def _need_adjust_camera(self, detected_result: MaskedBBOX):
        """camera framing for better view"""
        H, W, _ = self.observations.rgb.shape
        xmin = 0
        xmax = W - 1
        ymin = 0
        ymax = H - 1
        xmass_center = []

        for bbox, _, _ in detected_result:
            xmin = max(xmin, bbox[0])
            xmax = min(xmax, bbox[2])
            ymin = max(ymin, bbox[1])
            ymax = min(ymax, bbox[3])
            xmass_center.append((bbox[0] + bbox[2]) / 2)

        xmass_center = sum(xmass_center) / len(xmass_center)

        too_left = xmin < 5
        too_right = xmax > W - 5

        rot_num = abs(
            int((W / 2 - xmass_center) / W * self.config.SIMULATOR.RGB_SENSOR.HFOV)
            // self.turn_angle
        )

        # if too small, then adjust camera one step
        if (ymax - ymin) * (xmax - xmin) < 400:
            rot_num = 1
        if (
            too_left and self.camera_flag != "too_right" and rot_num
        ):  # avoid oscillation
            self.camera_flag = "too_left"
            logger.info("  too left, adjust camera...")
            for _ in range(rot_num):
                self.step(action=TURN_LEFT)
            return True
        elif too_right and self.camera_flag != "too_left" and rot_num:
            self.camera_flag = "too_right"
            logger.info("  too right, adjust camera...")
            for _ in range(rot_num):
                self.step(action=TURN_RIGHT)
            return True
        else:
            self.camera_flag = "center"
        return False

    def egoview_plan(self, maskbbox: MaskedBBOX) -> List[ObjectWayPoint]:
        ## TODO: map into larger area by vlmap
        det_waypts: List[ObjectWayPoint] = []
        for mbbox in maskbbox:
            goal_area = self.fbe.get_goal_area(mbbox[-1])
            if goal_area is None:
                continue
            cen_x = goal_area.x
            cen_z = goal_area.z

            navigable_mask = self.navigable_mask
            if self.use_gt_pose:
                navigable_mask = np.logical_and(
                    navigable_mask, self.fbe.occu_maping.map == OccupancyMapping.FREE
                )
            navigable_mask = vis.get_largest_connected_area(navigable_mask)

            wall_mask = self.fbe.occu_maping.map == OccupancyMapping.WALL
            # if the object is too high or too low, then the view point shoule be a little far away
            if abs(goal_area.height - self.agent_height) > 0.7:
                threshold = 25
                if len(goal_area.contour) <= 10:
                    threshold = 30
            elif abs(goal_area.height - self.agent_height) > 0.4:
                threshold = 20
                if len(goal_area.contour) <= 10:
                    threshold = 25
            else:
                threshold = 14
            logger.info(
                f"[EgoViewPlan] set threshold: {threshold} for the egodetected object"
            )
            grd_view = PointPlanner.plan_view_point_with_contour(
                cen_x,
                cen_z,
                navigable_mask,
                wall_mask,
                goal_area.contour,
                threshold=threshold,
            )
            grd_goal = Agent2DPose(x=cen_x, z=cen_z, t=0)
            m = re.search(r"\((.*?)\)", mbbox[1])
            if m:
                score = float(m.group(1))
            else:
                score = None

            det_waypts.append(
                ObjectWayPoint(
                    viewpt=grd_view,
                    goalpt=grd_goal,
                    contour=goal_area.contour,
                    height=goal_area.height,
                    score=score,
                )
            )

        for pt in det_waypts:
            pt.type = "ego_view"

        return self._remove_negative_waypts(det_waypts, self.negmem_results)
