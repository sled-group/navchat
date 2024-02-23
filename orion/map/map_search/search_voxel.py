import copy
from typing import List, Optional

import cv2
import numpy as np

from orion import logger
from orion.abstract.pose import Agent2DPose, ObjectWayPoint
from orion.config.my_config import VLMAP_QUERY_LIST_BASE, CLIPConfig, LsegConfig, torch
from orion.map.map_search.search_base import MapSearch
from orion.navigation.waypoint_planner import PointPlanner
from orion.perception.extractor.clipbase import CLIPBase


class VLMapSearch(MapSearch):
    # We suppose the map is fixed
    def __init__(
        self,
        clipcfg: CLIPConfig = LsegConfig(),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.clip_model = CLIPBase(cfg=clipcfg)

        text_feat = self.clip_model.encode_text(
            VLMAP_QUERY_LIST_BASE
        )  # [num_query, feat_dim]
        if isinstance(text_feat, torch.Tensor):
            text_feat = text_feat.cpu().numpy()
        self.text_feat = text_feat

        self.feat_dim = self.feat_values.shape[-1]
        assert (
            self.feat_dim == self.text_feat.shape[-1]
        ), f"text-vision feat dim not match {self.feat_dim} vs {self.text_feat.shape[-1]}"

    def query(self, text_list: List[str]):
        if isinstance(text_list, str):
            text_list = [text_list]
        query_list = copy.deepcopy(VLMAP_QUERY_LIST_BASE)
        for s in text_list:
            if s not in query_list:
                query_list.append(s)
        text_feat: np.ndarray = self.clip_model.encode_text(
            query_list
        )  # [num_query, feat_dim]
        if isinstance(text_feat, torch.Tensor):
            text_feat = text_feat.cpu().numpy()
        indices = self.indices  # [num_vxl, 3]
        vision_feat = self.feat_values  # [num_vxl, feat_dim]
        map_shape = self._3dshape  # (num_z, num_x, num_y, feat_dim)

        similarity = np.matmul(vision_feat, text_feat.T)  # [num_vxl, num_query]
        prediction = np.argmax(similarity, axis=1)  # [num_vxl]

        # get matching results for bird-eye view
        # Warning: If two objects are overlapped vertically, only the higher one will be matched
        predict_map = self.get_BEV_map(
            indices, np.expand_dims(prediction, axis=-1), (*map_shape[:3], 1)
        )
        predict_map = np.squeeze(predict_map, axis=-1).astype(np.uint8)

        return predict_map, query_list

    def plan(
        self,
        tgt_obj: str,
        query_list: List[str],
        predict_map: np.ndarray,
        navigatable_mask: Optional[np.ndarray] = None,
        wall_mask: Optional[np.ndarray] = None,
        show=False,
    ) -> List[ObjectWayPoint]:
        # In this function, we assume the predict_map_crop is already cropped
        # the returned coordinate are also cropped
        query_id = query_list.index(tgt_obj)
        logger.info(f"query_id, {query_id}, {tgt_obj}")

        grayimg = np.zeros(shape=predict_map.shape, dtype=np.uint8)
        grayimg[predict_map == query_id] = 255
        if np.sum(grayimg) == 0:
            return []  # no any objects found

        kernel = np.ones((1, 1), np.uint8)
        grayimg = cv2.dilate(grayimg, kernel, iterations=4)
        grayimg = cv2.erode(grayimg, kernel, iterations=2)
        contours, _ = cv2.findContours(
            grayimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            logger.info("no contours found")
            return []

        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        all_possible_tgts = []  # (cen_x, cen_z, area)
        for idx, c in enumerate(sorted_contours):
            # if cv2.contourArea(c) > 5.0:
            #     logger.info(f"contour area, {cv2.contourArea(c)}")
            M = cv2.moments(c)
            if idx == 0:
                if M["m00"] < 5.0:
                    logger.info("the object may too small to be detected.")
                    return []
                elif M["m00"] > 2200.0:
                    logger.info("the object may too large to be detected.")
                    return []
                else:
                    M_largest = copy.deepcopy(M)
                    if tgt_obj in [
                        "table",
                        "chair",
                        "cabinet",
                        "shelf",
                    ]:  # These are hacking
                        min_area = min(0.5 * M_largest["m00"], 35.0)
                    else:
                        min_area = 0.3 * M_largest["m00"]
                    cen_x = int(M["m10"] / M["m00"])
                    cen_z = int(M["m01"] / M["m00"])
                    all_possible_tgts.append((cen_x, cen_z, c))
                    # logger.info(f"centroid, {cen_x}, {cen_z}")
                    continue

            if M["m00"] < min_area:
                continue  # too small
            cen_x = int(M["m10"] / M["m00"])
            cen_z = int(M["m01"] / M["m00"])

            if any(
                np.sqrt((_cx - cen_x) ** 2 + (_cz - cen_z) ** 2) <= 3.0
                for _cx, _cz, _ in all_possible_tgts
            ):
                continue  # too close to previous recored centroids
            else:
                all_possible_tgts.append((cen_x, cen_z, c))
                # logger.info(f"centroid,{cen_x}, {cen_z}")

        if navigatable_mask is None:
            # use the floor as navigatable area
            navigatable_mask = predict_map == VLMAP_QUERY_LIST_BASE.index("floor")
            # cv2.imshow("navigatable_mask_crop", navigatable_mask_crop.astype(np.uint8)*255)
            # cv2.waitKey(0)
        if wall_mask is None:
            # use the wall as wall mask
            wall_mask = predict_map == VLMAP_QUERY_LIST_BASE.index("wall")
            # erode the wall mask
            # kernel = np.ones((1, 1), np.uint8)
            # wall_mask_crop = cv2.erode(wall_mask_crop.astype(np.uint8), kernel, iterations=1)
            # cv2.imshow("wall_mask_crop", wall_mask_crop.astype(np.uint8)*255)
            # cv2.waitKey(0)

        if show:
            colorimg = cv2.cvtColor(grayimg, cv2.COLOR_GRAY2BGR)
            colorimg[wall_mask] = [122, 0, 122]
            colorimg[navigatable_mask == 1] = [122, 122, 122]
        output = []
        for cen_x, cen_z, contour in all_possible_tgts:
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
            goalpt = Agent2DPose(cen_x, cen_z)

            # # change the coordinate to the full map
            # _goalpt = Agent2DPose(cen_x + self.xmin, cen_z + self.zmin)
            # _viewpt = (
            #     Agent2DPose(viewpt.x + self.xmin, viewpt.z + self.zmin)
            #     if viewpt is not None
            #     else None
            # )
            # _contour = contour + np.array([self.xmin, self.zmin])
            output.append(ObjectWayPoint(viewpt=viewpt, goalpt=goalpt, contour=contour))
            # logger.info("tgtpt", (cen_x, cen_z), "viewpt", viewpt)
            if show and viewpt is not None:
                cv2.circle(
                    colorimg, (viewpt.x, viewpt.z), 3, (0, 255, 255), -1
                )  # draw view point

        if show:
            cv2.imshow("vlmap_prediction", colorimg)
            cv2.waitKey(0)
        return output
