"""
Adapted from VLmap repo. https://github.com/vlmaps/vlmaps.
"""

from typing import Union

import clip
import numpy as np
import torch

from orion.abstract.interfaces import TextQueries
from orion.abstract.perception import PerceptionModule
from orion.config.my_config import LsegConfig
from orion.perception.extractor.lseg_module import LSegEncDecNet


class LSegExtractor(PerceptionModule):
    def __init__(self, cfg: LsegConfig = LsegConfig()):
        model = LSegEncDecNet(
            arch_option=0, block_depth=0, activation="lrelu", visualize=False
        )

        model_state_dict = model.state_dict()
        pretrained_state_dict = torch.load(cfg.ckpt_path)
        pretrained_state_dict = {
            k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()
        }
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(pretrained_state_dict)

        model.eval()
        model = model.to(cfg.device)
        self.model = model

        self.feat_dim = self.model.out_c
        self.device = cfg.device
        self.cfg = cfg

    @torch.no_grad()
    def predict(self, rgb: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(rgb, np.ndarray):
            rgb = np.expand_dims(rgb, axis=0)
        else:
            rgb = torch.unsqueeze(rgb, dim=0)
        outputs = self.model.encode(rgb)
        return outputs[0].permute(1, 2, 0)  # [H, W, D]

    @torch.no_grad()
    def encode_text(self, text_list: TextQueries) -> torch.Tensor:
        if isinstance(text_list, list):
            text_list = TextQueries(prompts=text_list)
        text = clip.tokenize(text_list.prompts).to(self.device)
        text_features = self.model.clip_pretrained.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.float()
        return text_features
