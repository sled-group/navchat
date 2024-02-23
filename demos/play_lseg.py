import torch
import cv2

from orion.utils.visulization import plot_pixel_feature_match
from orion.utils.file_load import load_image
from orion.config.my_config import LsegConfig
from orion.perception.extractor.clipbase import CLIPBase
from orion.abstract.interfaces import TextQueries
from orion.config.my_config import CLIPConfig_vitB32_openai, VLMAP_QUERY_LIST_BASE
from orion.perception.extractor.lseg_extractor import LSegExtractor

# # Test LSegExtractor
lseg_extractor = LSegExtractor(cfg=LsegConfig())


input_image = load_image("demos/demo.jpg")
# NB: has to change size to 480x640 !!!!
# Otherwise, the Lseg will not work
input_image = cv2.resize(input_image, (640, 480))

clip_extractor = CLIPBase(CLIPConfig_vitB32_openai(device="cpu"))

text_list = VLMAP_QUERY_LIST_BASE + ["dog", "cat", "grass", "tree"]
text_feat = clip_extractor.encode_text(TextQueries(text_list))
text_feat = text_feat.cpu().numpy()
print("text feature size: ", text_feat.shape)

with torch.no_grad():
    pixel_feat = lseg_extractor.predict(input_image).cpu().numpy()
    print("img feature size:", pixel_feat.shape)

plot_pixel_feature_match(
    pixel_feat, text_feat, text_list, save_path="demos/demo_lseg_output.jpg"
)
