### file loading ###
import cv2
import numpy as np

from orion.abstract.pose import Agent3DPose


def get_floor_set_str(floor_set):
    a, b = floor_set
    if a < 0:
        a = "B{}".format(-a)
    else:
        a = "U{}".format(a)
    if b < 0:
        b = "B{}".format(-b)
    else:
        b = "U{}".format(b)
    return "{}_{}".format(a, b)


def load_image(rgb_path):
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb


def load_depth(depth_filepath):
    with open(depth_filepath, "rb") as f:
        depth = np.load(f)
    if len(depth.shape) == 3:
        depth = depth.squeeze()
    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / 1000.0
    return depth


def load_semantic(semantic_filepath, obj2cls_dic):
    with open(semantic_filepath, "rb") as f:
        semantic = np.load(f)
    if len(semantic.shape) == 3:
        semantic = semantic.squeeze()
    semantic = np.asarray(semantic).astype(np.int32)
    semantic = cvt_sem_id_2_cls_id(semantic, obj2cls_dic)
    return semantic


def cvt_sem_id_2_cls_id(semantic: np.ndarray, obj2cls: dict):
    h, w = semantic.shape
    semantic = semantic.flatten()
    u, inv = np.unique(semantic, return_inverse=True)
    return np.array([obj2cls[x][0] for x in u])[inv].reshape((h, w))


def load_obj2cls_dict(filepath):
    obj2cls_dic = {}
    label_dic = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = line.split(":")
            obj_id = int(row[0])
            cls_id = int(row[1].split(",")[0].strip())
            cls_name = row[1].split(",")[1].strip()
            obj2cls_dic[obj_id] = (cls_id, cls_name)
            label_dic[cls_id] = cls_name
    label_dic = dict(sorted(label_dic.items(), key=lambda x: x[0]))
    return obj2cls_dic, label_dic


def load_pose(pose_filepath):
    with open(pose_filepath, "r") as f:
        line = f.readline()
        return Agent3DPose.from_str(line)
