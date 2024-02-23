import json
import os
import warnings

# Grounding DINO
import groundingdino.datasets.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from PIL import Image
# segment anything
from segment_anything import SamPredictor, build_sam

from orion import logger
from orion.abstract.interfaces import Observations, TextQuery
from orion.abstract.perception import DetectionModule, MaskedBBOX
from orion.config.my_config import GroundingDINOConfig

warnings.filterwarnings("ignore")


def load_image(image):
    if isinstance(image, str) and os.path.exists(image):
        image_path = image
        image_pil = Image.open(image_path).convert("RGB")  # load image
    else:
        image_pil = Image.fromarray(image)

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    _ = model.eval()
    return model


def get_grounding_output(
    model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"
):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(
            logit > text_threshold, tokenized, tokenlizer
        )
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, "mask.jpg"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )

    json_data = [{"value": value, "label": "background"}]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split("(")
        logit = logit[:-1]  # the last is ')'
        json_data.append(
            {
                "value": value,
                "label": name,
                "logit": float(logit),
                "box": box.numpy().tolist(),
            }
        )
    with open(os.path.join(output_dir, "mask.json"), "w") as f:
        json.dump(json_data, f)


class GroundingSAM(DetectionModule):
    def __init__(self, cfg=GroundingDINOConfig()):
        self.cfg = cfg

        # load model
        self.model = load_model(
            self.cfg.config_file, self.cfg.grounded_checkpoint, device=self.cfg.device
        )

        # initialize SAM
        self.predictor = SamPredictor(
            build_sam(checkpoint=self.cfg.sam_checkpoint).to(self.cfg.device)
        )

    def predict(self, rgb: np.ndarray, txt: TextQuery) -> MaskedBBOX:
        """text prompt should be a sentence or multiple words separated by
        ' . '. tgt_object should be a single noun word
        """
        # load image

        txt_prompt: str = txt.prompt.lower() if txt.prompt else ""
        txt_object: str = txt.target.lower() if txt.target else ""

        image_pil, image = load_image(rgb)

        # run grounding dino model
        boxes_filt, pred_phrases = get_grounding_output(
            self.model,
            image,
            txt_prompt,
            self.cfg.box_threshold,
            self.cfg.text_threshold,
            device=self.cfg.device,
        )

        if all(txt_object not in p for p in pred_phrases):
            return MaskedBBOX(False, [], [], [])

        self.predictor.set_image(rgb)
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()

        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            boxes_filt, rgb.shape[:2]
        ).to(self.cfg.device)

        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.cfg.device),
            multimask_output=False,
        )

        bboxes = boxes_filt.numpy().astype(np.int32)
        texts = pred_phrases
        masks = masks.cpu().numpy()

        tuple_list = []
        for bbox, text, mask in zip(bboxes, texts, masks):
            if txt_object in text:
                tuple_list.append((bbox, text, mask))

        return MaskedBBOX.from_tuple_list(True, tuple_list)
