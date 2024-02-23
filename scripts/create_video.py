import numpy as np
import cv2

import sys
import os
import imageio
import numpy as np
import cv2
import tqdm

from habitat_sim.utils.common import d3_40_colors_rgb


has_gpu = True  # @param {type: "boolean"}
codec = "h264"
if has_gpu:
    codec = "h264_nvenc"


def load_depth(depth_filepath):
    with open(depth_filepath, "rb") as f:
        depth = np.load(f)
    return depth


def get_fast_video_writer(video_file: str, fps: int = 60):
    if (
        "google.colab" in sys.modules
        and os.path.splitext(video_file)[-1] == ".mp4"
        and os.environ.get("IMAGEIO_FFMPEG_EXE") == "/usr/bin/ffmpeg"
    ):
        # USE GPU Accelerated Hardware Encoding
        writer = imageio.get_writer(
            video_file,
            fps=fps,
            codec=codec,
            mode="I",
            bitrate="1000k",
            format="FFMPEG",
            ffmpeg_log_level="info",
            quality=10,
            output_params=["-minrate", "500k", "-maxrate", "5000k"],
        )
    else:
        # Use software encoding
        writer = imageio.get_writer(video_file, fps=fps)
    return writer


def create_video(data_dir: str, fps: int = 30):
    rgb_dir = os.path.join(data_dir, "rgb")
    rgb_list = sorted(
        os.listdir(rgb_dir), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]

    depth_dir = os.path.join(data_dir, "depth")
    depth_list = sorted(
        os.listdir(depth_dir), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    depth_list = [os.path.join(depth_dir, x) for x in depth_list]

    semantic_dir = os.path.join(data_dir, "semantic")
    semantic_list = sorted(
        os.listdir(semantic_dir), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    semantic_list = [os.path.join(semantic_dir, x) for x in semantic_list]

    assert len(rgb_list) == len(depth_list) == len(semantic_list)

    output_path = os.path.join(data_dir, "recording_video.mp4")
    out_writer = get_fast_video_writer(output_path, fps=fps)

    pbar = tqdm.tqdm(total=len(rgb_list), position=0, leave=True)
    for i, (rgb_path, depth_path, semantic_path) in enumerate(
        list(zip(rgb_list, depth_list, semantic_list))
    ):
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        depth = np.load(open(depth_path, "rb"))
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000
        depth_vis = (depth / 10 * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        semantic = np.load(open(semantic_path, "rb"))
        semantic_color = d3_40_colors_rgb[semantic.squeeze() % 40]
        output_im = np.concatenate((rgb, depth_color, semantic_color), axis=1)
        out_writer.append_data(output_im)
        pbar.update(1)
    out_writer.close()


if __name__ == "__main__":
    create_video("data/experiments/fbetest_4ok3usBNeis_B1_U1/recordings")
