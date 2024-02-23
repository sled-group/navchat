import cv2
from orion.perception.detector.clipgradcam import CLIPGradCAM


from orion.utils.file_load import load_image

input_image = load_image("demos/demo.jpg")

clipgradcam = CLIPGradCAM()
returnpt = clipgradcam.predict(input_image, "dog")

# plot the centroid into the image
import matplotlib.pyplot as plt

plt.imshow(input_image)
plt.scatter(returnpt[0], returnpt[1], c="r", s=100)
plt.savefig(
    "demos/demo_gradcam_output.jpg",
    bbox_inches="tight",
    dpi=300,
    pad_inches=0.0,
)
