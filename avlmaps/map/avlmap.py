from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import open3d as o3d
import numpy as np
from scipy.ndimage import distance_transform_edt
from omegaconf import DictConfig


from avlmaps.dataloader.habitat_dataloader import VLMapsDataloaderHabitat
from avlmaps.map import VLMap, SoundMap, AreaMap, VisualMap
from avlmaps.utils.visualize_utils import get_heatmap_from_mask_3d, convert_heatmap_to_rgb


class AVLMap:
    def __init__(self, config: DictConfig, data_dir: str = ""):
        self.config = config

        print("######################## Start Initialization ###########################")
        print("Initializing VLMap")
        self.vlmap = VLMap(config.map_config, data_dir=data_dir)

        print("Initializing SoundMap")
        self.sound_map = SoundMap(
            data_dir, config.sound_config, config.sound_data_collect_params, is_ambiguous=False, is_real=False
        )

        print("Initializing AreaMap")
        self.area_map = AreaMap(data_dir)

        print("Initializing VisualMap")
        self.visual_map = VisualMap(config.map_config, data_dir)
        print("######################## Initialization Done ###########################")

    def create_map(self, data_dir: Union[Path, str]) -> None:
        self.vlmap.create_map(data_dir)

        self.area_map.create_map(data_dir)

        self.visual_map.create_and_load_map(data_dir)

        self.sound_map.create_sound_map(data_dir)

        return True

    def load_map(self, data_dir: str):
        self.vlmap.load_map(data_dir)
        self.area_map.load_map(data_dir)
        self.visual_map.create_and_load_map(data_dir)
        self.sound_map.load_sound_map(data_dir)
        self.dataloader = VLMapsDataloaderHabitat(data_dir, self.config.map_config, self.vlmap)
        return True

    def visualize_heatmap(self, heatmap: np.ndarray, transparency: float = 0.5) -> o3d.geometry.PointCloud:
        heat_rgb = convert_heatmap_to_rgb(heatmap, self.vlmap.grid_rgb, transparency)

        grid_rgb = heat_rgb / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.vlmap.grid_pos)
        pcd.colors = o3d.utility.Vector3dVector(grid_rgb)
        return pcd

    def index_object(self, object_name: str, init_categories: List[str] = None, decay_rate: float = 0.1) -> np.ndarray:
        if init_categories is not None:
            self.vlmap.init_categories(init_categories[1:-1])
            mask = self.vlmap.index_map(object_name, with_init_cat=True)
        else:
            mask = self.vlmap.index_map(object_name, with_init_cat=False)
        heatmap = get_heatmap_from_mask_3d(
            self.vlmap.grid_pos, mask, cell_size=self.config.params.cs, decay_rate=decay_rate
        )
        return heatmap

    def index_area_2d(self, area_name: str, decay_rate: float = 0.1) -> np.ndarray:
        dist_map = np.zeros(self.vlmap.occupied_ids.shape[:2], dtype=np.float32)
        scores = self.area_map.index_map(area_name, with_init_cat=False)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        robot_pose_list = self.area_map.robot_pose_list

        for i, tf_hab in enumerate(robot_pose_list):
            tmp_dist_map = np.zeros_like(dist_map, dtype=np.float32)
            self.dataloader.from_habitat_tf(tf_hab)
            row, col, deg = self.dataloader.to_full_map_pose()
            if row < 0 or row >= dist_map.shape[0] or col < 0 or col >= dist_map.shape[1]:
                continue
            s = scores[i]
            tmp_dist_map[row, col] = s
            dists = distance_transform_edt(tmp_dist_map == 0)
            tmp = np.ones_like(dists) * s - (dists * decay_rate)
            tmp_dist_map = np.clip(tmp, 0, 1)
            dist_map = np.where(dist_map > tmp_dist_map, dist_map, tmp_dist_map)

        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))
        return dist_map

    def index_area(self, area_name: str, decay_rate: float = 0.1) -> np.ndarray:
        heatmap_2d = self.index_area_2d(area_name, decay_rate)
        heatmap_3d = np.zeros(self.vlmap.grid_pos.shape[0], dtype=np.float32)
        rows, cols, heights = np.where(self.vlmap.occupied_ids != -1)

        for row, col, heigh in zip(rows, cols, heights):
            id = self.vlmap.occupied_ids[row, col, heigh]
            heatmap_3d[id] = heatmap_2d[row, col]

        return heatmap_3d

    def index_sound_2d(self, sound_name: str, decay_rate: float = 0.01) -> np.ndarray:
        probabilities, locations_list = self.sound_map.get_distribution_and_locations(sound_name)
        dist_map = np.zeros(self.vlmap.occupied_ids.shape[:2], dtype=np.float32)
        print(dist_map.shape)
        for loc_i, locations in enumerate(locations_list):
            tmp_dist_map = np.zeros_like(dist_map, dtype=np.float32)
            for location in locations:
                tf_hab = np.eye(4)
                tf_hab[:3, 3] = location
                self.dataloader.from_habitat_tf(tf_hab)
                row, col, deg = self.dataloader.to_full_map_pose()
                tmp_dist_map[row, col] = probabilities[loc_i]

            con = probabilities[loc_i]
            dists = distance_transform_edt(tmp_dist_map == 0)
            reduct = con * dists * decay_rate
            tmp = np.ones_like(tmp_dist_map) * con - reduct
            tmp_dist_map = np.where(tmp < 0, np.zeros_like(tmp), tmp)
            dist_map += tmp_dist_map

        dist_map = (dist_map - np.min(dist_map)) / (np.max(dist_map) - np.min(dist_map))

        return dist_map

    def index_sound(self, sound_name: str, decay_rate: float = 0.01) -> np.ndarray:
        heatmap_2d = self.index_sound_2d(sound_name, decay_rate)
        heatmap_3d = np.zeros(self.vlmap.grid_pos.shape[0], dtype=np.float32)
        rows, cols, heights = np.where(self.vlmap.occupied_ids != -1)

        for row, col, heigh in zip(rows, cols, heights):
            id = self.vlmap.occupied_ids[row, col, heigh]
            heatmap_3d[id] = heatmap_2d[row, col]

        return heatmap_3d

    def index_image(
        self, image: np.ndarray, query_cam_intrinsics: np.ndarray = None, decay_rate: float = 0.01
    ) -> np.ndarray:
        query_cam_tf, query_base_tf = self.visual_map.localize_image(
            image, query_cam_intrinsic_mat=query_cam_intrinsics
        )
        self.dataloader.from_habitat_tf(query_base_tf)
        row, col, deg = self.dataloader.to_full_map_pose()
        height = self.config.map_config.pose_info.camera_height / self.config.params.cs
        con = 1.0
        pos = np.array([row, col, height])
        sim_mat = np.zeros((self.vlmap.grid_pos.shape[0], 1))
        dists = np.linalg.norm((self.vlmap.grid_pos - pos)[:, :2], axis=1)
        sim = np.clip(con - decay_rate * dists, 0, 1)
        sim_mat[:, 0] = sim
        sim = np.max(sim_mat, axis=1)

        return sim.flatten()
