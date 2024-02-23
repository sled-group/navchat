import collections
import copy
import json
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple
from attr import define, field
import numpy as np
import cv2
import numpy as np
import networkx as nx

from orion.config.my_config import *
from orion.utils import visulization as vis
from orion.navigation.waypoint_planner import PointPlanner

from orion import logger

import random

random.seed(1)


@define
class Instance:
    # a class for target objects from semantic map

    name: str
    id: int  # name+id is unique in the semantic map
    center: Tuple[int, int]  # center in the topdown view
    contour: Optional[np.ndarray]
    mass: float  # contour area
    room_id: Optional[str] = None  # room id, e.g. bedroom_1

    object_desc: Optional[
        List[str]
    ] = None  # alias name, additional info for the object
    room_desc: Optional[str] = None  # alias name, additional info for the room

    nearby_obj: Optional[List[str]] = None

    explain: Optional[str] = None  # explain the object
    same_goal: Optional[
        Tuple[str]
    ] = None  # tuple of object id. reach any of it consider success
    type: Optional[str] = None  # object type  1. big  2. ambiguous  3. small
    attr: Optional[str] = None  # special attribute

    def __attrs_post_init__(self):
        if self.object_desc is None:
            self.object_desc = []
        if self.nearby_obj is None:
            self.nearby_obj = []

    def __str__(self):
        return_str = f"{self.id}, room: {self.room_id}, {self.center}, {self.mass}"
        if self.object_desc:
            return_str += f"\n  object_desc: {self.object_desc}"
        if self.room_desc:
            return_str += f"\n  room_desc: {self.room_desc}"
        if self.nearby_obj:
            return_str += f"\n  nearby_obj: {self.nearby_obj}"
        if self.explain:
            return_str += f"\n  explain: {self.explain}"
        if self.same_goal:
            return_str += f"\n  same_goal: {self.same_goal}"
        if self.type:
            return_str += f"\n  type: {self.type}"
        if self.attr:
            return_str += f"\n  attr: {self.attr}"

        return return_str

    def __repr__(self):
        return self.__str__()


class TopologicalGraph:
    def __init__(
        self, load_sparse_map_path, label_dic, map_querier, nearby_dist_thres=20
    ):
        self.dir = os.path.dirname(load_sparse_map_path)
        self.label_dic = label_dic

        self.map_querier = map_querier
        self.mapshape = self.map_querier._3dshape
        self.indices = self.map_querier.indices  # [num_vxl, 3]
        self.gt_values = self.map_querier.gt_values  # [num_vxl]

        self.wall_mask = np.zeros(
            shape=(self.mapshape[0], self.mapshape[1]), dtype=np.uint8
        )

        self.G = nx.Graph()
        self.instance_dict: Dict[str, Instance] = {}
        self.subgraphs = []
        self.room_count = {"bedroom": 0, "bathroom": 0, "kitchen": 0, "living room": 0}

        self.nearby_dist_thres = nearby_dist_thres

        self._build_graph()

        self.all_instance_dict = copy.deepcopy(self.instance_dict)

        self.person_nams = NAMES.copy()
        self.company_names = COMPANIES.copy()
        self.year_choice = YEARS.copy()
        self.material_choice = MATERIALS.copy()

    def _build_nodes(self):
        total_id = np.unique(self.map_querier.gt_values).tolist()
        label_dic = {
            k: v
            for k, v in self.label_dic.items()
            if k in total_id and v not in ["misc", "machine"]
        }
        name_dic = dict([(v, k) for k, v in label_dic.items()])

        for name in name_dic:
            object_id = name_dic[name]
            gt_mask = self.gt_values == object_id
            predict_map = self.map_querier.get_BEV_map(
                self.indices,
                np.expand_dims(gt_mask, axis=-1),
                map_shape=[*self.mapshape, 1],
            )

            predict_map = predict_map.squeeze()
            if self._wall_filter(name):
                self.wall_mask = np.logical_or(predict_map, self.wall_mask)
                continue
            if self._object_filter(name):
                continue

            predict_map = predict_map.astype(np.uint8)
            kernel = np.ones((1, 1), np.uint8)
            grayimg = cv2.dilate(predict_map, kernel, iterations=4)
            grayimg = cv2.erode(predict_map, kernel, iterations=2)
            contours, _ = cv2.findContours(
                grayimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(contours) == 0:
                continue

            sorted_contours = sorted(
                contours, key=lambda x: cv2.contourArea(x), reverse=True
            )
            largest_contour_area = cv2.contourArea(sorted_contours[0])

            filtered_contours = []
            if largest_contour_area < 13:
                continue
            for c in sorted_contours:
                area = cv2.contourArea(c)
                if area > max(
                    min(largest_contour_area / 2, 22), largest_contour_area * 0.1
                ):
                    filtered_contours.append(c)

            for i, c in enumerate(filtered_contours):
                node_id = name + f"_{i}"
                self.G.add_node(node_id)
                M = cv2.moments(c)
                centroid_x = int(M["m10"] / M["m00"])
                centroid_y = int(M["m01"] / M["m00"])
                instance = Instance(
                    name=name,
                    id=node_id,
                    center=(centroid_x, centroid_y),
                    contour=c,
                    mass=cv2.contourArea(c),
                )

                self.instance_dict[node_id] = instance

    def _build_edges(self):
        keys = list(self.instance_dict.keys())
        for idx1 in range(len(keys)):
            for idx2 in range(len(keys)):
                if idx1 == idx2:
                    continue
                k1 = keys[idx1]
                k2 = keys[idx2]
                ins1 = self.instance_dict[k1]
                ins2 = self.instance_dict[k2]
                dist = self.instance_dist(ins1, ins2)

                _, is_in_view = PointPlanner.line_search(
                    x0=ins1.center[0],
                    y0=ins1.center[1],
                    x1=ins2.center[0],
                    y1=ins2.center[1],
                    wall_mask=self.wall_mask,
                    stop_at_wall=True,
                    object_contours=[ins1.contour, ins2.contour],
                )
                # wall_mask = self.wall_mask.astype(np.uint8)
                # kernel = np.ones((1, 1), np.uint8)
                # grayimg = cv2.dilate(wall_mask, kernel, iterations=4)
                # grayimg = cv2.erode(wall_mask, kernel, iterations=2)

                # vis.plot_uint8img_with_plt(self.wall_mask.astype(np.uint8)*255, title="wall_mask", crop=True)
                # input()

                if is_in_view:  # in the same room possibly
                    self.G.add_edge(k1, k2, weight=dist)

    def _build_graph(self):
        self._build_nodes()
        self._build_edges()

        # Find connected components in the graph
        connected_components = list(nx.connected_components(self.G))

        # Helper function to check if two nodes are directly connected by an edge
        def are_nodes_directly_connected(node1, node2):
            if node1 == node2:
                return True
            return node2 in self.G[node1]

        # Iterate through each connected component and split them into subgraphs
        for component in connected_components:
            isolated_nodes = list(copy.deepcopy(component))
            while len(isolated_nodes) > 0:
                node1 = isolated_nodes.pop()
                subgraph = set()
                subgraph.add(node1)

                for n in component:
                    if n not in isolated_nodes:
                        continue
                    if all(
                        are_nodes_directly_connected(n, node2) for node2 in subgraph
                    ):
                        subgraph.add(n)
                        isolated_nodes.remove(n)

                self.subgraphs.append(
                    sorted(
                        list(subgraph),
                        key=lambda x: self.instance_dict[x].mass,
                        reverse=True,
                    )
                )

        # Assign rooms to some subgraphs
        for subgraph in self.subgraphs:
            if any(re.match(r"bed_\d+", name) for name in subgraph):
                room_name = "bedroom"
                self.room_count["bedroom"] += 1
                for n in subgraph:
                    self.instance_dict[n].room_id = (
                        room_name + f"_{self.room_count[room_name]}"
                    )

            elif any(
                re.search(r"(bathroom|toilet|bathtub)", name) for name in subgraph
            ):
                room_name = "bathroom"
                self.room_count["bathroom"] += 1
                for n in subgraph:
                    self.instance_dict[n].room_id = (
                        room_name + f"_{self.room_count[room_name]}"
                    )

            elif (
                any(re.search(r"(couch|sofa|carpet|tv)_\d+", name) for name in subgraph)
                and self.room_count["living room"] == 0
            ):
                room_name = "living room"
                self.room_count["living room"] += 1
                for n in subgraph:
                    self.instance_dict[n].room_id = (
                        room_name + f"_{self.room_count[room_name]}"
                    )

            elif (
                any(re.search(r"^kitchen", name) for name in subgraph)
                and self.room_count["kitchen"] == 0
            ):
                room_name = "kitchen"
                self.room_count["kitchen"] += 1
                for n in subgraph:
                    self.instance_dict[n].room_id = (
                        room_name + f"_{self.room_count[room_name]}"
                    )

        # add nearby objects
        for node in self.G.nodes:
            self.instance_dict[node].nearby_obj = self.get_sorted_neighbors(
                node, self.nearby_dist_thres
            )

    def visulization(self):
        topdown_rgb = self.map_querier.get_BEV_map(
            indices=self.map_querier.indices,
            values=self.map_querier.rgb_values,
            map_shape=[*self.mapshape, 3],
        )

        topdown_rgb[self.wall_mask == 1] = [255, 255, 255]

        for subgraph in self.subgraphs:
            random_rgb = [np.random.randint(120, 255) for _ in range(3)]

            for node in subgraph:
                instance = self.instance_dict[node]
                cv2.drawContours(topdown_rgb, [instance.contour], 0, random_rgb, -1)

        vis.plot_uint8img_with_plt(
            topdown_rgb,
            title="cluster",
            crop=True,
            save_path=os.path.join(os.path.dirname(self.dir), "cluster.png"),
            save=True,
        )

    @property
    def all_names(self):
        return set(re.sub(r"_\d+", "", k) for k in self.instance_dict.keys())

    def retrieve_objects_by_name(self, name_str):
        if name_str not in self.all_names:
            logger.warning(f"{name_str} not in the graph")
            return []
        return [
            k for k, v in self.instance_dict.items() if re.match(rf"^{name_str}_\d+", k)
        ]

    def retrieve_objects_by_roomid(self, room_id):
        room, room_num = room_id.split("_")
        if room not in ROOMS or not (0 < int(room_num) <= self.room_count[room]):
            logger.warning(f"{room_id} not in the graph")
            return []
        return [k for k, v in self.instance_dict.items() if room_id == v.room_id]

    def get_largest_objects_each_category(self, mass_thres=0):
        obj_mass_tuplelist = []
        print("self.all_names", self.all_names)
        for n in self.all_names:
            objs = self.retrieve_objects_by_name(n)
            print("objs", objs)
            objs = sorted(objs, key=lambda x: self.instance_dict[x].mass, reverse=True)
            if len(objs) > 0:
                obj_mass_tuplelist.append((n, self.instance_dict[objs[0]].mass))
        obj_mass_tuplelist = sorted(
            obj_mass_tuplelist, key=lambda x: x[1], reverse=True
        )
        return [
            n for n, m in obj_mass_tuplelist if m > mass_thres
        ]  # return list of names

    def get_sorted_objects(self):
        output = []
        for node in self.G.nodes:
            if node in self.instance_dict:
                output.append((node, self.instance_dict[node].mass))
        output = sorted(output, key=lambda x: x[1], reverse=True)
        output = [n for n, m in output]  # return list of sorted obj_id
        # return dict {name:[obj_id]}
        dic = collections.defaultdict(list)
        for n in output:
            dic[re.sub(r"_\d+", "", n)].append(n)
        return dic, output

    def suggest_instance_goal(self, mass_thres=50, add_persona=False):
        if add_persona:  # consider bedrooms, bathrooms belongs to different people
            if self.room_count["bedroom"] > 1:
                for i in range(1, self.room_count["bedroom"] + 1):
                    obj_in_room = self.retrieve_objects_by_roomid(f"bedroom_{i}")
                    person = self.person_nams.pop(0)
                    for oid in obj_in_room:
                        self.instance_dict[oid].room_desc = f"{person}'s bedroom"
                        self.instance_dict[oid].object_desc.append(
                            f"{person}'s {self.instance_dict[oid].name}"
                        )

            if self.room_count["bathroom"] > 1:
                for i in range(1, self.room_count["bathroom"] + 1):
                    obj_in_room = self.retrieve_objects_by_roomid(f"bathroom_{i}")
                    person = self.person_nams.pop(0)
                    # for oid in obj_in_room:
                    #     self.instance_dict[oid].room_desc = f"{person}'s bathroom"
                    #     self.instance_dict[oid].object_desc.append(
                    #         f"{person}'s {self.instance_dict[oid].name}"
                    #     )

        # find out distincive objects
        large_objs = self.get_largest_objects_each_category(mass_thres)
        obj_dic, _ = self.get_sorted_objects()
        instance_goals = {}  # {name: {objid: set()}}
        for n, l in obj_dic.items():
            if n not in large_objs:
                continue
            if len(l) == 1:  # if only one instance, add it
                instance_goals[n] = {l[0]: set([l[0]])}
                if add_persona:
                    ins = self.instance_dict[l[0]]
                    if ins.room_desc is None:
                        desc = self._get_obj_desc(ins)
                        if desc:
                            ins.object_desc.append(desc)
            else:  # if multiple instances, determine distincive ones
                instance_goals[n] = {}
                mapping_dic = collections.defaultdict(set)
                for oid in l:
                    ins = self.instance_dict[oid]
                    for k, v in mapping_dic.items():
                        if any(
                            self._is_similar(ins, self.instance_dict[idx]) for idx in v
                        ):
                            v.add(oid)
                            break
                    else:
                        mapping_dic[oid].add(oid)

                for k, v in mapping_dic.items():
                    instance_goals[n][k] = v
                    if add_persona:
                        for idx in v:
                            ins = self.instance_dict[idx]
                            if ins.room_desc is None:
                                desc = self._get_obj_desc(ins)
                                if desc:
                                    ins.object_desc.append(desc)

        return instance_goals

    def _get_obj_desc(self, ins: Instance):
        rand = random.random()
        if rand < 0.8:
            if len(self.company_names) > 0 and len(self.year_choice) > 0:
                company = self.company_names.pop(0)
                year = self.year_choice.pop(0)
                return f"bought from {company} {year}"
            if len(self.material_choice) > 0:
                material = self.material_choice.pop(0)
                return f"made of {material}"
            if len(self.person_nams) > 0:
                person = self.person_nams.pop(0)
                return f"{person}'s {ins.name}"
        return None

    def _is_similar(self, ins1: Instance, ins2: Instance):
        # similar objects will be considered as the same object, reach any of them is considered success
        # same_dist = self.instance_dist(ins1, ins2) < self.nearby_dist_thres
        if ins1.room_id is not None and ins2.room_id is not None:
            same_room = ins1.room_id == ins2.room_id
        else:
            same_room = self.instance_dist(ins1, ins2) < self.nearby_dist_thres
        same_name = ins1.name == ins2.name
        same_nearyby = len(
            set([re.sub(r"_\d+$", "", o) for o in ins1.nearby_obj[:2]])
            & set([re.sub(r"_\d+$", "", o) for o in ins2.nearby_obj[:2]])
        )
        return same_room and same_name and same_nearyby

    def is_neighbor(self, node1, node2, dist_thres=20):
        if node1 not in self.G:
            logger.warning(f"{node1} not in the graph")
            return False
        if node2 not in self.G:
            logger.warning(f"{node2} not in the graph")
            return False
        return node2 in self.G[node1] and self.G[node1][node2]["weight"] < dist_thres

    def get_sorted_neighbors(self, node, dist_thres=20):
        if node not in self.G:
            logger.warning(f"{node} not in the graph")
            return []
        neighbors = self.G[node]
        neighbors = [n for n in neighbors if self.G[node][n]["weight"] < dist_thres]
        neighbors = sorted(
            neighbors, key=lambda x: self.G[node][x]["weight"], reverse=True
        )
        return neighbors

    def get_cloest_object(self, node):
        neighbors = self.get_sorted_neighbors(node)
        if neighbors:
            return neighbors[0]
        else:
            return None

    @staticmethod
    def contour_dist(c1, c2):
        # c1, c2: np.ndarray  [N, 1, 2]
        # find the nearest distance between two contours along all the points

        c1 = c1.squeeze()
        c2 = c2.squeeze()

        dist = np.linalg.norm(c1[:, None, :] - c2[None, :, :], axis=-1)
        return np.min(dist)

    @staticmethod
    def point_dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def instance_dist(self, ins1: Instance, ins2: Instance):
        return min(
            self.contour_dist(ins1.contour, ins2.contour),
            self.point_dist(ins1.center, ins2.center),
        )

    @staticmethod
    def point_contour_dist(p, c):
        c = c.squeeze()
        dist = np.linalg.norm(c - np.array(p), axis=-1)
        return np.min(dist)

    @staticmethod
    def _wall_filter(s):
        for k in WALL_OBJECTS:
            if k == s:
                return True
        return False

    @staticmethod
    def _object_filter(s):
        # those objects will not be considered in the graph
        for o in FILTER_OBJECTS:
            if o in s:
                return True
        return False

    def load_instance_data(self, usr_goals, all_objects):
        # load from json goal file
        room_mapping = {}
        self.instance_dict = {}  # goal instance

        for k, v in usr_goals.items():
            if k == "room_info":
                for room_name, desc in v.items():
                    desc_list = [i for i in desc.split("|")]
                    for i, d in enumerate(desc_list):
                        if d == "shared":
                            room_mapping[f"{room_name}_{i+1}"] = f"{room_name}"
                        else:
                            room_mapping[f"{room_name}_{i+1}"] = f"{d}'s {room_name}"
            else:
                if "same_goal" in v:
                    same_goal = tuple([i for i in v["same_goal"].split("|") if i != ""])
                else:
                    same_goal = (k,)
                
                # assert all(i in self.all_instance_dict for i in same_goal)
                for ii in same_goal:
                    if ii not in self.all_instance_dict:
                        logger.warning(f"{ii} not in the graph")

                if k in all_objects:
                    name = all_objects[k]["name"]
                    center = tuple(all_objects[k]["center"])
                    contour = np.asarray(all_objects[k]["contour"])
                    mass = all_objects[k]["mass"]
                else:
                    name = re.sub(r"_\d+$", "", k)
                    base_dic = json.loads(v["base"])
                    center = tuple(base_dic["center"])
                    mass = base_dic["mass"]
                    contour = np.array(base_dic["center"]).reshape(-1, 1, 2)

                self.instance_dict[k] = Instance(
                    name=name,
                    id=k,
                    center=tuple(center),
                    contour=contour,
                    mass=mass,
                    room_id=v["room_id"],
                    room_desc=room_mapping[v["room_id"]]
                    if v["room_id"] in room_mapping
                    else None,
                    object_desc=[i for i in v["object_desc"].split("|") if i != ""],
                    nearby_obj=[i for i in v["nearby_obj"].split("|") if i != ""],
                    explain=v["explain"],
                    same_goal=same_goal,
                    type=v["type"],
                    attr=v["attr"],
                )
        
        for k in self.instance_dict:
            if k not in self.all_instance_dict:
                self.all_instance_dict[k] = copy.deepcopy(self.instance_dict[k])