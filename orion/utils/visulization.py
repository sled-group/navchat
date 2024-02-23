from typing import List, Optional

import cv2
import matplotlib.patches as mpatches
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from orion import logger


def get_contour_points(pos, origin=(0, 0), size=20):
    x, y, o = pos
    pt1 = (int(x) + origin[0], int(y) + origin[1])
    pt2 = (
        int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1],
    )
    pt3 = (int(x + size * np.cos(o)) + origin[0], int(y + size * np.sin(o)) + origin[1])
    pt4 = (
        int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
        int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1],
    )

    return np.array([pt1, pt2, pt3, pt4])


def plot_uint8img_with_plt(array, title="", crop=True, save=False, save_path=None):
    array = array.astype(np.uint8)
    if len(array.shape) == 3 and array.shape[2] == 3:
        if crop:
            z_indices, x_indices = np.where(array.sum(axis=-1) > 0)
            xmin = np.min(x_indices)
            xmax = np.max(x_indices)
            zmin = np.min(z_indices)
            zmax = np.max(z_indices)
            array = array[zmin : zmax + 1, xmin : xmax + 1, :]
            logger.info(f"{xmin}, {xmax}, {zmin}, {zmax}")
            logger.info(f"{array.shape}")

        image = Image.fromarray(array)
        plt.figure(dpi=120)
        plt.imshow(image)
        plt.title(title)
        if save:
            if save_path is not None:
                plt.savefig(save_path)
            else:
                plt.savefig(title + ".png")
        else:
            plt.show()
    elif len(array.shape) == 2 or (len(array.shape) == 3 and array.shape[2] == 1):
        array = array.squeeze()
        image = Image.fromarray(array)
        plt.figure(figsize=(8, 8), dpi=120)
        plt.imshow(image, cmap="gray")
        plt.title(title)
        if save:
            if save_path is not None:
                plt.savefig(save_path)
            else:
                plt.savefig(title + ".png")
        else:
            plt.show()
    else:
        raise ValueError("Not supported image shape")


def get_new_mask_pallete(
    npimg, new_palette, out_label_flag=False, label_dic=None, ignore_ids_list=[-1, 0]
):
    """Get image color pallete for visualizing masks"""
    # put colormap
    out_img = Image.fromarray(npimg.squeeze().astype("uint8"))
    out_img.putpalette(new_palette)

    if out_label_flag:
        assert label_dic is not None
        u_index = np.unique(npimg)
        patches = []
        for i, index in enumerate(u_index):
            if index in ignore_ids_list:
                continue
            label = label_dic[index]
            cur_color = [
                new_palette[index * 3] / 255.0,
                new_palette[index * 3 + 1] / 255.0,
                new_palette[index * 3 + 2] / 255.0,
            ]
            red_patch = mpatches.Patch(color=cur_color, label=label)
            patches.append(red_patch)
    return out_img, patches


def get_auto_pallete(num_cls, make_wall_black=True):
    # by default, the first three classes are "other", "wall", "floor"
    n = num_cls
    pallete = [0] * (n * 3)

    for j in range(0, n):
        lab = j
        pallete[j * 3 + 0] = 0
        pallete[j * 3 + 1] = 0
        pallete[j * 3 + 2] = 0
        i = 0
        while lab > 0:
            pallete[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            pallete[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            pallete[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i = i + 1
            lab >>= 3
    if make_wall_black:
        pallete[3:6] = [0, 0, 0]
    return pallete


def plot_pixel_feature_match(
    img_feat: np.ndarray,
    text_feat: np.ndarray,
    labels: List[str],
    save_path: Optional[str] = None,
):
    # text_feat: [n, 512], img_feat: [h, w, 512]
    # input_text: [n]
    # plot text_feat and img_feat in 2D space
    # plot text_feat and img_feat in 2D space
    if isinstance(text_feat, torch.Tensor):
        text_feat = text_feat.cpu().numpy()
    if isinstance(img_feat, torch.Tensor):
        img_feat = img_feat.cpu().numpy()
    predict = np.einsum("hwc,nc->hwn", img_feat, text_feat)
    predict = predict.argmax(axis=-1)
    new_palette = get_auto_pallete(len(labels), make_wall_black=False)
    mask, patches = get_new_mask_pallete(
        predict, new_palette, out_label_flag=True, label_dic=labels, ignore_ids_list=[]
    )
    seg = mask.convert("RGBA")
    # show legend
    plt.figure()
    plt.legend(
        handles=patches, loc="upper right", bbox_to_anchor=(1.0, 1), prop={"size": 10}
    )
    plt.axis("off")
    plt.imshow(seg)
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_BEV_semantic_map(
    predict_map: np.ndarray,
    nomap_mask: np.ndarray,
    labels: List[str],
    title="BEV_predict_map",
    save=False,
    save_path=None,
):
    from orion.config.my_config import VLMAP_QUERY_LIST_BASE

    new_palette = get_auto_pallete(len(labels), make_wall_black=True)
    outimg, patches = get_new_mask_pallete(
        predict_map, new_palette, out_label_flag=True, label_dic=labels
    )
    seg = outimg.convert("RGBA")
    seg = np.array(seg)
    seg[nomap_mask] = [225, 225, 225, 255]
    floor_mask = predict_map == VLMAP_QUERY_LIST_BASE.index("floor")
    seg[floor_mask] = [225, 225, 225, 255]
    other_mask = predict_map == VLMAP_QUERY_LIST_BASE.index("other")
    seg[other_mask] = [225, 225, 225, 255]
    seg = Image.fromarray(seg)
    plt.figure(figsize=(10, 6), dpi=120)
    plt.legend(
        handles=patches, loc="upper left", bbox_to_anchor=(1.0, 1), prop={"size": 10}
    )
    plt.axis("off")
    plt.title(title)
    plt.imshow(seg)
    if save:
        if save_path is not None:
            plt.savefig(save_path)
            logger.info(f"Save to {save_path}")
        else:
            plt.savefig(title + ".png")
            logger.info(f"Save to {title}.png")
    else:
        plt.show()


def contour_overlap(num_grid: int, contour1: np.ndarray, contour2: np.ndarray):
    area1 = cv2.contourArea(contour1)
    mask1 = np.zeros((num_grid, num_grid), dtype=np.uint8)
    mask1 = cv2.drawContours(mask1, [contour1], 0, 1, thickness=cv2.FILLED)

    area2 = cv2.contourArea(contour2)
    mask2 = np.zeros((num_grid, num_grid), dtype=np.uint8)
    mask2 = cv2.drawContours(mask2, [contour2], 0, 1, thickness=cv2.FILLED)

    intersection_mask = cv2.bitwise_and(mask1, mask2)
    contours, _ = cv2.findContours(
        intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) == 0:
        return 0
    intersection_area = cv2.contourArea(contours[0])

    if area1 == 0 or area2 == 0:
        center1 = np.mean(contour1, axis=0).squeeze()
        center2 = np.mean(contour2, axis=0).squeeze()
        if np.linalg.norm(center1 - center2) < 10:
            return 1
        else:
            return 0

    overlap_ratio = intersection_area / min(area1, area2)

    return overlap_ratio


def contour_merge(num_grid, contour1: np.ndarray, contour2: np.ndarray):
    mask1 = np.zeros((num_grid, num_grid), dtype=np.uint8)
    mask1 = cv2.drawContours(mask1, [contour1], 0, 1, thickness=cv2.FILLED)

    mask2 = np.zeros((num_grid, num_grid), dtype=np.uint8)
    mask2 = cv2.drawContours(mask2, [contour2], 0, 1, thickness=cv2.FILLED)

    union_mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(
        union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(contours) == 0:
        return None
    return contours[0]


def get_largest_connected_area(mask: np.ndarray, erosion_iter=1):
    # get the largest connected component
    dtype = mask.dtype
    mask = mask.astype(np.uint8)

    # erode the mask, this can avoid bad points
    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=erosion_iter)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        canvas = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(canvas, [contours[0]], -1, 1, thickness=cv2.FILLED)

        mask = mask & canvas
        mask = mask.astype(dtype)

        return mask
    else:
        return mask.astype(dtype)
