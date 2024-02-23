"""
Adapted from ConceptFusion repo. https://github.com/concept-fusion/concept-fusion
"""


from typing import Union

import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from orion.abstract.perception import PerceptionModule
from orion.config.my_config import (CLIPConfig_vitL14_datacomp,
                                    ConceptFusionConfig)
from orion.perception.extractor.clipbase import CLIPBase


class ConceptFusionExtractor(PerceptionModule):
    def __init__(
        self, cfg: ConceptFusionConfig = ConceptFusionConfig(), height=480, width=640
    ):
        self.height = height
        self.width = width
        self.sam = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_ckpt_path)
        self.sam.to(device=cfg.device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=8,
            pred_iou_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

        self.clip_model = CLIPBase(CLIPConfig_vitL14_datacomp())
        self.device = cfg.device
        self.feat_dim = self.clip_model.feat_dim

    @torch.no_grad()
    def predict(self, rgb: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        # Extracting SAM masks..
        masks = self.mask_generator.generate(rgb)  # around 4s in cuda for 480x640

        with torch.cuda.amp.autocast():
            # Extracting global CLIP features
            global_feat = self.clip_model.encode_image(rgb)
            global_feat /= global_feat.norm(dim=-1, keepdim=True)  # (1, h, w, feat_dim)
        global_feat = torch.nn.functional.normalize(global_feat, dim=-1)
        feat_dim = global_feat.shape[-1]
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

        feat_per_roi = []
        roi_nonzero_inds = []
        similarity_scores = []
        for maskidx in range(len(masks)):
            try:
                _x, _y, _w, _h = tuple(masks[maskidx]["bbox"])  # xywh bounding box
                seg = masks[maskidx]["segmentation"]
                nonzero_inds = torch.argwhere(
                    torch.from_numpy(masks[maskidx]["segmentation"])
                )
                # Note: Image is (H, W, 3). In SAM output, y coords are along height, x along width
                img_roi = rgb[_y : _y + _h, _x : _x + _w, :]
                roifeat = self.clip_model.encode_image(img_roi)
                roifeat = torch.nn.functional.normalize(roifeat, dim=-1)
            except:
                roifeat = global_feat.clone().detach()
            feat_per_roi.append(roifeat)
            roi_nonzero_inds.append(nonzero_inds)
            _sim = cosine_similarity(global_feat, roifeat)
            similarity_scores.append(_sim)

        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)
        outfeat = torch.zeros(self.height, self.width, feat_dim, dtype=torch.half)
        for maskidx in range(len(masks)):
            _weighted_feat = (
                softmax_scores[maskidx] * global_feat
                + (1 - softmax_scores[maskidx]) * feat_per_roi[maskidx]
            )
            _weighted_feat = torch.nn.functional.normalize(_weighted_feat, dim=-1)
            outfeat[
                roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
            ] += (_weighted_feat[0].detach().cpu().half())
            outfeat[
                roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
            ] = torch.nn.functional.normalize(
                outfeat[
                    roi_nonzero_inds[maskidx][:, 0], roi_nonzero_inds[maskidx][:, 1]
                ].float(),
                dim=-1,
            ).half()

        outfeat = outfeat.unsqueeze(0).float()
        outfeat = torch.nn.functional.normalize(outfeat, dim=-1)
        outfeat = outfeat[0].half()  # --> H, W, feat_dim
        return outfeat.cpu()
