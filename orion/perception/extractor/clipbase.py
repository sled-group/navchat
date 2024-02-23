from typing import List, Union

import numpy as np
import open_clip
import torch
from PIL import Image

from orion.abstract.interfaces import TextQueries
from orion.config.my_config import CLIPConfig


class CLIPBase:
    """use clip text encoder to get vector representation of text,
    other vision backbone to get pixel-wise vector representation of image,
    two vectors should are in the same semantic space
    """

    def __init__(self, cfg: CLIPConfig):
        self.device = cfg.device
        self.clip_version = cfg.clip_version
        self.openclip_pretained = cfg.openclip_pretained
        self.feat_dim = {"ViT-B-32": 512, "ViT-B-16": 512, "ViT-L-14": 768}[
            cfg.clip_version
        ]
        (
            self.clip_model,
            _,
            self.clip_preprocess,
        ) = open_clip.create_model_and_transforms(
            self.clip_version, pretrained=self.openclip_pretained
        )
        self.tokenizer = open_clip.get_tokenizer(self.clip_version)
        self.clip_model.to(self.device)
        self.cfg = cfg

    def encode_text(self, txts: Union[List[str], TextQueries]) -> torch.Tensor:
        if isinstance(txts, list):
            txts = TextQueries(txts)
        tok = self.tokenizer(txts.prompts).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.clip_model.encode_text(tok)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def encode_image(self, image: Union[np.ndarray, Image.Image, str]) -> torch.Tensor:
        # single image -> single vec
        if isinstance(image, np.ndarray):
            assert len(image.shape) == 3, "image should be [h, w, c]"
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image)

        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def score(self, image_feat: torch.Tensor, text_feat: torch.Tensor) -> np.ndarray:
        return (100.0 * image_feat @ text_feat.T).softmax(dim=-1).cpu().numpy()

    def predict(self, rgb: np.ndarray, txts: TextQueries):
        image_feat = self.encode_image(rgb)
        text_feat = self.encode_text(txts)
        text_probs = self.score(image_feat, text_feat)
        return text_probs
