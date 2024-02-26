from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import gdown

from tqdm import tqdm
import clip
import cv2
import torchvision.transforms as transforms
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.ndimage import binary_closing, binary_dilation, gaussian_filter
import torch

from avlmaps.utils.clip_utils import get_text_feats_multiple_templates
from avlmaps.utils.visualize_utils import pool_3d_label_to_2d

from avlmaps.map.vlmap_builder_multi_floor import VLMapBuilderMultiFloor
from avlmaps.utils.mapping_utils import load_3d_map
from avlmaps.map.map import Map
from avlmaps.utils.index_utils import find_similar_category_id, get_segment_islands_pos, get_dynamic_obstacles_map_3d
from avlmaps.utils.clip_utils import get_lseg_score


class VLMapMultiFloor(Map):
    def __init__(self, map_config: DictConfig, data_dir: str = ""):
        super().__init__(map_config, data_dir=data_dir)
        self.scores_mat = None
        self.categories = None

    def _setup_paths(self, data_dir: Union[Path, str]) -> None:
        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / "rgb"
        self.depth_dir = self.data_dir / "depth"
        self.semantic_dir = self.data_dir / "semantic"
        self.pose_dir = self.data_dir / "pose"
        try:
            self.rgb_paths = sorted(self.rgb_dir.glob("*.png"))
            self.depth_paths = sorted(self.depth_dir.glob("*.png"))
            self.semantic_paths = sorted(self.semantic_dir.glob("*.npy"))
            self.pose_paths = sorted(self.pose_dir.glob("*.txt"))
        except FileNotFoundError as e:
            print(e)

    def create_map(self, data_dir: Union[Path, str]) -> None:
        print(f"Creating map for scene at: ", data_dir)
        self._setup_paths(data_dir)
        self.map_builder = VLMapBuilderMultiFloor(
            self.data_dir,
            self.map_config,
            self.pose_paths,
            self.rgb_paths,
            self.depth_paths,
            self.base2cam_tf,
            self.base_transform,
        )
        if self.map_config.pose_info.pose_type == "mobile_base":
            self.map_builder.create_mobile_base_map()
        elif self.map_config.pose_info.pose_type == "camera":
            self.map_builder.create_camera_map()
        elif self.map_config.pose_info.pose_type == "global":
            self.map_builder.create_global_map()

    def load_map(self, data_dir: str) -> bool:
        self._setup_paths(data_dir)
        self.map_save_path = Path(data_dir) / "vlmap_multi_floor" / "vlmaps_multi_floor.h5df"
        if not self.map_save_path.exists():
            print("Loading VLMap failed because the file doesn't exist.")
            return False
        (
            self.mapped_iter_list,
            self.grid_feat,
            self.grid_pos,
            self.weight,
            self.occupied_ids,
            self.grid_rgb,
            self.pcd_min,
            self.pcd_max,
            self.cs,
        ) = VLMapBuilderMultiFloor.load_3d_map(self.map_save_path)

        return True

    def _init_clip(self, clip_version="ViT-B/32"):
        if hasattr(self, "clip_model"):
            print("clip model is already initialized")
            return
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.clip_version = clip_version
        self.clip_feat_dim = {
            "RN50": 1024,
            "RN101": 512,
            "RN50x4": 640,
            "RN50x16": 768,
            "RN50x64": 1024,
            "ViT-B/32": 512,
            "ViT-B/16": 512,
            "ViT-L/14": 768,
        }[self.clip_version]
        print("Loading CLIP model...")
        self.clip_model, self.preprocess = clip.load(self.clip_version)  # clip.available_models()
        self.clip_model.to(self.device).eval()

    def init_categories(self, categories: List[str]) -> np.ndarray:
        self.categories = categories
        self.scores_mat = get_lseg_score(
            self.clip_model,
            self.categories,
            self.grid_feat,
            self.clip_feat_dim,
            use_multiple_templates=True,
            add_other=True,
        )  # score for name and other
        return self.scores_mat

    def index_map(self, language_desc: str, with_init_cat: bool = True):
        if with_init_cat and self.scores_mat is not None and self.categories is not None:
            cat_id = find_similar_category_id(language_desc, self.categories)
            scores_mat = self.scores_mat
        else:
            if with_init_cat:
                raise Exception(
                    "Categories are not preloaded. Call init_categories(categories: List[str]) to initialize categories."
                )
            scores_mat = get_lseg_score(
                self.clip_model,
                [language_desc],
                self.grid_feat,
                self.clip_feat_dim,
                use_multiple_templates=True,
                add_other=True,
            )  # score for name and other
            cat_id = 0

        max_ids = np.argmax(scores_mat, axis=1)
        mask = max_ids == cat_id
        return mask

    def customize_obstacle_map(
        self,
        potential_obstacle_names: List[str],
        obstacle_names: List[str],
        vis: bool = False,
    ):
        if self.obstacles_cropped is None and self.obstacles_map is None:
            self.generate_obstacle_map()
        if not hasattr(self, "clip_model"):
            print("init_clip in customize obstacle map")
            self._init_clip()

        self.obstacles_new_cropped = get_dynamic_obstacles_map_3d(
            self.clip_model,
            self.obstacles_cropped,
            self.map_config.potential_obstacle_names,
            self.map_config.obstacle_names,
            self.grid_feat,
            self.grid_pos,
            self.rmin,
            self.cmin,
            self.clip_feat_dim,
            vis=vis,
        )
        self.obstacles_new_cropped = Map._dilate_map(
            self.obstacles_new_cropped == 0,
            self.map_config.dilate_iter,
            self.map_config.gaussian_sigma,
        )
        self.obstacles_new_cropped = self.obstacles_new_cropped == 0

    def get_pos(self, name: str) -> Tuple[List[List[int]], List[List[float]], List[np.ndarray], Any]:
        """
        Get the contours, centers, and bbox list of a certain category
        on a full map
        """
        assert self.categories
        pc_mask = self.index_map(name, with_init_cat=True)
        mask_2d = pool_3d_label_to_2d(pc_mask, self.grid_pos, self.gs)
        mask_2d = mask_2d[self.rmin : self.rmax + 1, self.cmin : self.cmax + 1]

        foreground = binary_closing(mask_2d, iterations=3)
        foreground = gaussian_filter(foreground.astype(float), sigma=0.8, truncate=3)
        foreground = foreground > 0.5
        foreground = binary_dilation(foreground)

        contours, centers, bbox_list, _ = get_segment_islands_pos(foreground, 1)

        # whole map position
        for i in range(len(contours)):
            centers[i][0] += self.rmin
            centers[i][1] += self.cmin
            bbox_list[i][0] += self.rmin
            bbox_list[i][1] += self.rmin
            bbox_list[i][2] += self.cmin
            bbox_list[i][3] += self.cmin
            for j in range(len(contours[i])):
                contours[i][j, 0] += self.rmin
                contours[i][j, 1] += self.cmin

        return contours, centers, bbox_list
