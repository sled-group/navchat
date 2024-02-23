from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
import torch

from orion import logger
from orion.abstract.interfaces import TextQuery


class PerceptionModule(ABC):
    @abstractmethod
    def predict(
        self, rgb: np.ndarray, txt: TextQuery
    ) -> Union["MaskedBBOX", float, np.ndarray, torch.Tensor]:
        """
        single image prediction
        """


class DetectionModule(PerceptionModule):
    """img -> bboxes"""

    @abstractmethod
    def predict(self, rgb: np.ndarray, txt: TextQuery) -> "MaskedBBOX":
        pass


class ExtractorModule(PerceptionModule):
    """img -> feature vector"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_dim = None

    @abstractmethod
    def predict(
        self, rgb: np.ndarray, txt: Optional[TextQuery] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        pass


@dataclass
class MaskedBBOX:
    flag: bool
    bboxes: List[Any]  # (x1, y1, x2, y2)
    texts: List[str]
    masks: List[np.ndarray]

    def __bool__(self):
        return self.flag

    def __len__(self):
        return len(self.bboxes)

    def __iter__(self):
        return iter(zip(self.bboxes, self.texts, self.masks))

    def __getitem__(self, idx):
        return self.bboxes[idx], self.texts[idx], self.masks[idx]

    @classmethod
    def from_tuple_list(cls, flag, tuple_list):
        if len(tuple_list) == 0:
            return cls(flag, [], [], [])
        tuple_list = sorted(
            tuple_list, key=lambda x: cls._bbox_area(x[0]), reverse=True
        )
        return cls(flag, *zip(*tuple_list))

    @staticmethod
    def _bbox_area(bbox):
        if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            return 0
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def de_duplication(self):
        if len(self) in [0, 1]:
            return self

        mask_idx = []
        for i in range(len(self)):
            for j in range(i + 1, len(self)):
                bbox_i = self.bboxes[i]
                bbox_j = self.bboxes[j]
                bbox_j_area = self._bbox_area(bbox_j)
                intersect_box = [
                    max(bbox_i[0], bbox_j[0]),
                    max(bbox_i[1], bbox_j[1]),
                    min(bbox_i[2], bbox_j[2]),
                    min(bbox_i[3], bbox_j[3]),
                ]
                intersect_area = self._bbox_area(intersect_box)

                mask_i: np.ndarray = self.masks[i].squeeze()
                mask_j: np.ndarray = self.masks[j].squeeze()

                intersect_mask = np.logical_and(mask_i, mask_j)

                if (
                    intersect_area / bbox_j_area > 0.8
                    or np.sum(intersect_mask) / np.sum(mask_j) > 0.9
                ):
                    logger.info(
                        f"[MBBOX] de_duplication: " "{self.bboxes[i]}, {self.bboxes[j]}"
                    )
                    mask_idx.append(j)

        tuple_list = []
        logger.info(f"[MBBOX] before: {len(self)}")
        for i in range(len(self)):
            if i not in mask_idx:
                tuple_list.append((self.bboxes[i], self.texts[i], self.masks[i]))
        logger.info(f"[MBBOX] after: {len(tuple_list)}")
        return MaskedBBOX.from_tuple_list(self.flag, tuple_list)
