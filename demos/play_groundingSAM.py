import cv2
from orion.abstract.interfaces import TextQuery
from orion.perception.detector.groundingSAM import GroundingSAM, show_mask, show_box

import matplotlib.pyplot as plt

from orion.utils.file_load import load_image

input_image = load_image("demos/demo.jpg")

text_prompt = TextQuery(prompt="dog on the grass", target="dog")


plt.figure(figsize=(10, 10))
plt.imshow(input_image)

groundsam = GroundingSAM()

mmboxes = groundsam.predict(input_image, text_prompt)
for mask in mmboxes.masks:
    show_mask(mask, plt.gca(), random_color=True)
for box, label in zip(mmboxes.bboxes, mmboxes.texts):
    show_box(box, plt.gca(), label)

plt.axis("off")
plt.savefig(
    "demos/demo_groundedsam_output.jpg",
    bbox_inches="tight",
    dpi=300,
    pad_inches=0.0,
)
