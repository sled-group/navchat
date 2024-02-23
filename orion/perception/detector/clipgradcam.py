import cv2
import numpy as np
import torch
from PIL import Image

import orion.perception.detector.gradcam.CLIP.clip as clip


class CLIPGradCAM:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device, jit=False
        )

    def interpret(self, image, texts, start_layer=-1):
        batch_size = texts.shape[0]
        images = image.repeat(batch_size, 1, 1, 1)
        logits_per_image, logits_per_text = self.model(images, texts)
        probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
        index = [i for i in range(batch_size)]
        one_hot = np.zeros(
            (logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32
        )
        one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * logits_per_image)
        self.model.zero_grad()

        image_attn_blocks = list(
            dict(self.model.visual.transformer.resblocks.named_children()).values()
        )

        if start_layer == -1:
            # calculate index of last layer
            start_layer = len(image_attn_blocks) - 1

        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(
            num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype
        ).to(self.device)
        R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
        for i, blk in enumerate(image_attn_blocks):
            if i < start_layer:
                continue
            grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[
                0
            ].detach()
            cam = blk.attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        image_relevance = R[:, 0, 1:]

        dim = int(image_relevance.numel() ** 0.5)
        image_relevance = image_relevance.reshape(1, 1, dim, dim)
        image_relevance = torch.nn.functional.interpolate(
            image_relevance, size=224, mode="bilinear"
        )
        image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
        image_relevance = (image_relevance - image_relevance.min()) / (
            image_relevance.max() - image_relevance.min()
        )

        return self.find_centroid(image_relevance)

    def find_centroid(self, object_mask):
        us, vs = np.where(object_mask > 0.99)
        if len(us) == 0:
            return None

        mean_u = np.mean(us)
        mean_v = np.mean(vs)
        index = np.argmin((us - mean_u) ** 2 + (vs - mean_v) ** 2, axis=None)
        y, x = us[index], vs[index]
        return (x, y)  # (7,215)

    def predict(self, rgb: np.ndarray, txt: str):
        img = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        texts = [txt]
        text = clip.tokenize(texts).to(self.device)
        pt = self.interpret(image=img, texts=text)
        if pt is None:
            return None
        else:
            # resize back to original image size
            x, y = pt
            x = int(pt[0] * rgb.shape[1] / 224)
            x = max(min(x, rgb.shape[1] - 1), 0)
            y = int(pt[1] * rgb.shape[0] / 224)
            y = max(min(y, rgb.shape[0] - 1), 0)
            return (x, y)
