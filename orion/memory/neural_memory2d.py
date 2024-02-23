import os
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import torch
from orion import logger
from orion.abstract.memory import EpisodicMemory
from orion.abstract.pose import Agent2DPose, ObjectWayPoint
from orion.navigation.waypoint_planner import PointPlanner


class SparseMem2D:
    def __init__(self, feat_dim) -> None:
        # sparsed vectorized self.memory map
        self.indices = np.empty((0, 2), dtype=np.int32)
        self.values = np.empty((0, feat_dim), dtype=np.float32)
        # zx -> index
        self.indice_dic: Dict[Any, int] = {}
        # number of voxels in this grid
        self.grd_count: List[int] = []


class NeuralMemory2D(EpisodicMemory):
    """A 3D sparsed vectorized self.memory map"""

    def __init__(self, num_grid, feat_dim):
        self.num_grid = num_grid
        self.feat_dim = feat_dim

        self.positive_memory: SparseMem2D = SparseMem2D(feat_dim)
        self.negative_memory: SparseMem2D = SparseMem2D(feat_dim)

    def add(self, feat_val, grd_zxs):
        # feat_values: [K, feat_dim]
        # grd_zxs: [K, 2], zx
        if isinstance(feat_val, torch.Tensor):
            feat_val = feat_val.cpu().numpy()
        self._update_grd(feat_val, grd_zxs, self.positive_memory)

    def delete(self, feat_val, grd_zxs):
        if isinstance(feat_val, torch.Tensor):
            feat_val = feat_val.cpu().numpy()
        self._update_grd(feat_val, grd_zxs, self.negative_memory)

    def update(self, neg_feat_val, tgt_feat_val, grd_zxs):
        if neg_feat_val is not None:
            self.delete(neg_feat_val, grd_zxs)
        if tgt_feat_val is not None:
            self.add(tgt_feat_val, grd_zxs)

    def _update_grd(self, feat_val, grd_zxs, memory: SparseMem2D):
        feat_val = feat_val.reshape(-1, self.feat_dim)
        feat_val = np.mean(feat_val, axis=0)
        assert (
            len(feat_val) == self.feat_dim
        ), f"feat_val should be {self.feat_dim}, not {len(feat_val)}"
        for grd_zx in grd_zxs:
            self.is_in_grid(grd_zx)
            grd_zx_tuple = tuple(grd_zx.tolist())
            if grd_zx_tuple in memory.indice_dic:
                idx = memory.indice_dic[grd_zx_tuple]
                memory.values[idx] = (
                    feat_val + memory.values[idx] * memory.grd_count[idx]
                ) / (memory.grd_count[idx] + 1)
                memory.grd_count[idx] += 1
            else:
                memory.indice_dic[grd_zx_tuple] = len(memory.indices)
                memory.indices = np.append(memory.indices, [grd_zx], axis=0)
                memory.values = np.append(memory.values, [feat_val], axis=0)
                memory.grd_count.append(1)

    def retrieve(
        self, feat_emb, navigatable_mask, wall_mask
    ) -> Tuple[List[ObjectWayPoint], List[ObjectWayPoint]]:
        # find suitable area in postive memory
        # find non-suitable area in negative memory
        feat_emb = feat_emb.reshape(-1, self.feat_dim)  # [num_query, feat_dim]

        # find the most suitable area in positive memory
        logger.info("match the vector positive memory")
        vl_match_pos = self._vlmatch(self.positive_memory, feat_emb)

        # find the most non-suitable area in negative memory
        logger.info("match the vector negative memory")
        vl_match_neg = self._vlmatch(self.negative_memory, feat_emb)

        # logger.info("===plan on the vector positive memory===")
        output_pos: List[ObjectWayPoint] = self._plan(
            vl_match_pos, navigatable_mask, wall_mask
        )

        # logger.info("===plan on the vector negative memory===")
        output_neg: List[ObjectWayPoint] = self._plan(
            vl_match_neg, navigatable_mask, wall_mask
        )

        return output_pos, output_neg

    def retrieve_neg(
        self, feat_emb, navigatable_mask, wall_mask
    ) -> List[ObjectWayPoint]:
        feat_emb = feat_emb.reshape(-1, self.feat_dim)
        vl_match_neg = self._vlmatch(self.negative_memory, feat_emb)
        # logger.info("===plan on the vector negative memory===")
        output_neg: List[ObjectWayPoint] = self._plan(
            vl_match_neg, navigatable_mask, wall_mask
        )

        return output_neg

    def _vlmatch(self, memory: SparseMem2D, feat_emb):
        if isinstance(feat_emb, torch.Tensor):
            feat_emb = feat_emb.cpu().numpy()
        feat_emb = feat_emb.reshape(-1, self.feat_dim)
        vlmatch = np.zeros((self.num_grid, self.num_grid), dtype=np.uint8)
        if len(memory.indices) > 0:
            similarity = np.matmul(
                memory.values, feat_emb.T
            ).squeeze()  # [num_grd, num_query]
            max_similarity = max(np.max(similarity) - 0.03, 0.88)
            logger.info(
                f"max similarity: {max_similarity} | [{np.min(similarity)}, {np.max(similarity)}]"
            )
            prediction = similarity > max_similarity

            z = memory.indices[:, 0]
            x = memory.indices[:, 1]
            vlmatch[z[prediction], x[prediction]] = 255
        return vlmatch

    def _plan(
        self, match_map, navigatable_mask, wall_mask, show=False
    ) -> List[ObjectWayPoint]:
        grayimg = match_map.astype(np.uint8)
        kernel = np.ones((1, 1), np.uint8)
        grayimg = cv2.dilate(grayimg, kernel, iterations=2)
        # grayimg = cv2.erode(grayimg, kernel, iterations=2)
        contours, _ = cv2.findContours(
            grayimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            return []

        if show:
            colorimg = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2BGR)
            colorimg[wall_mask == 1] = [200, 200, 200]
            colorimg[navigatable_mask == 1] = [0, 122, 0]

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        output = []
        for contour in sorted_contours:
            M = cv2.moments(contour)
            if M["m00"] <= 2.0:
                continue
            cen_x = int(M["m10"] / M["m00"])
            cen_z = int(M["m01"] / M["m00"])
            if show:
                cv2.circle(
                    colorimg, (cen_x, cen_z), 3, (0, 0, 255), -1
                )  # draw target point
            # treat the center point as the goal
            # decide the radius according to the contour area

            viewpt = PointPlanner.plan_view_point_with_contour(
                cen_x=cen_x,
                cen_z=cen_z,
                navigable_mask=navigatable_mask,
                wall_mask=wall_mask,
                contour=contour,
            )

            # change the coordinate to the full map
            goalpt = Agent2DPose(cen_x, cen_z)
            output.append(ObjectWayPoint(viewpt=viewpt, goalpt=goalpt, contour=contour))
            # logger.info("tgtpt", (cen_x, cen_z), "viewpt", viewpt)
            if show and viewpt is not None:
                cv2.circle(
                    colorimg, (viewpt.x, viewpt.z), 3, (0, 255, 255), -1
                )  # draw view point

        if show:
            cv2.imshow("colorimg", colorimg)
            cv2.waitKey(0)
        return output

    def save(self, save_dir, suffix=""):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            logger.warning(f"{save_dir} already exists. (sparse map)")
        save_path = os.path.join(save_dir, f"sparse_postive_memory{suffix}.npz")
        np.savez(
            save_path,
            indices=self.positive_memory.indices,
            values=self.positive_memory.values,
            vxl_count=self.positive_memory.grd_count,
        )
        logger.info(f"{save_path} is saved. (sparse map)")

        save_path = os.path.join(save_dir, f"sparse_negative_memory{suffix}.npz")
        np.savez(
            save_path,
            indices=self.negative_memory.indices,
            values=self.negative_memory.values,
            vxl_count=self.negative_memory.grd_count,
        )
        logger.info(f"{save_path} is saved. (sparse map)")

    def load(self, load_dir):
        save_path = os.path.join(load_dir, "sparse_postive_memory.npz")
        if os.path.exists(save_path):
            data = np.load(save_path)
            self.positive_memory.indices = data["indices"]
            self.positive_memory.values = data["values"]
            self.positive_memory.grd_count = data["vxl_count"]
            for idx, grd_zx in enumerate(self.positive_memory.indices):
                self.positive_memory.indice_dic[tuple(grd_zx.tolist())] = idx
            logger.info(f"{save_path} is loaded. (sparse map)")
        else:
            logger.warning(f"{save_path} does not exist. (sparse map)")

        save_path = os.path.join(load_dir, "sparse_negative_memory.npz")
        if os.path.exists(save_path):
            data = np.load(save_path)
            self.negative_memory.indices = data["indices"]
            self.negative_memory.values = data["values"]
            self.negative_memory.grd_count = data["vxl_count"]
            for idx, grd_zx in enumerate(self.negative_memory.indices):
                self.negative_memory.indice_dic[tuple(grd_zx.tolist())] = idx
            logger.info(f"{save_path} is loaded. (sparse map)")
        else:
            logger.warning(f"{save_path} does not exist. (sparse map)")

    def is_in_grid(self, grd_zx):
        if grd_zx[0] < 0 or grd_zx[0] >= self.num_grid:
            logger.warning(f"grd_zx[0] out of range: {grd_zx[0]}")
            grd_zx[0] = np.clip(grd_zx[0], 0, self.num_grid - 1)
        if grd_zx[1] < 0 or grd_zx[1] >= self.num_grid:
            logger.warning(f"grd_zx[1] out of range: {grd_zx[1]}")
            grd_zx[1] = np.clip(grd_zx[1], 0, self.num_grid - 1)
