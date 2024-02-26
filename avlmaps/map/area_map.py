from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from tqdm import tqdm
import clip
import cv2
import numpy as np
from omegaconf import DictConfig
import torch

from avlmaps.utils.clip_utils import get_img_feats, get_text_feats
from avlmaps.utils.mapping_utils import save_clip_sparse_map, load_clip_sparse_map, cvt_pose_vec2tf
from avlmaps.utils.index_utils import find_similar_category_id


class AreaMap:
    def __init__(self, data_dir: str = "") -> None:
        self.clip_sparse_map = None
        self.robot_pose_list = None
        self._init_clip()
        if data_dir:
            self._setup_paths(data_dir)

    def _init_clip(self, clip_version="ViT-L/14"):
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

    def _setup_paths(self, data_dir: Union[Path, str]) -> None:
        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / "rgb"
        self.depth_dir = self.data_dir / "depth"
        self.pose_path = self.data_dir / "poses.txt"
        self.map_save_dir = self.data_dir / "area_map"
        os.makedirs(self.map_save_dir, exist_ok=True)
        try:
            self.rgb_paths = sorted(self.rgb_dir.glob("*.png"))
            self.depth_paths = sorted(self.depth_dir.glob("*.npy"))
        except FileNotFoundError as e:
            print(e)

    def create_map(self, data_dir: Union[Path, str]) -> None:
        data_dir = Path(data_dir)
        self._setup_paths(data_dir)  # setup self.map_save_dir and makedirs
        print(f"loading scene {data_dir}")

        self.base_poses = np.loadtxt(self.pose_path)

        clip_sparse_map_path = self.map_save_dir / f"clip_sparse_map.h5df"

        clip_sparse_map = np.zeros((len(self.rgb_paths), self.clip_feat_dim), dtype=np.float32)
        robot_pose_list = []

        pbar = tqdm(enumerate(zip(self.rgb_paths, self.base_poses)), total=len(self.rgb_paths))
        # load all images and depths and poses
        for iter_i, (rgb_path, base_posevec) in pbar:
            pbar.set_description(desc=f"Creating AreaMap Frame {iter_i:06}")

            bgr = cv2.imread(rgb_path.as_posix())
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            habitat_base_pose = cvt_pose_vec2tf(base_posevec)

            img_feats = get_img_feats(rgb, self.preprocess, self.clip_model)
            clip_sparse_map[iter_i] = img_feats.flatten()
            robot_pose_list.append(habitat_base_pose)
        self.clip_sparse_map = clip_sparse_map
        self.robot_pose_list = robot_pose_list
        save_clip_sparse_map(clip_sparse_map_path, clip_sparse_map, robot_pose_list)

    def load_map(self, data_dir: Union[Path, str]) -> None:
        self._setup_paths(data_dir)  # setup self.map_save_dir and makedirs
        clip_sparse_map_path = self.map_save_dir / f"clip_sparse_map.h5df"
        self.clip_sparse_map, self.robot_pose_list = load_clip_sparse_map(clip_sparse_map_path)

    def init_categories(self, categories: List[str]) -> np.ndarray:
        self.categories = categories
        text_feats = get_text_feats(categories, self.clip_model, self.clip_feat_dim)
        self.scores_mat = self.clip_sparse_map @ text_feats.T
        return self.scores_mat

    def index_map(self, language_desc: str, with_init_cat: bool = True):
        if with_init_cat and self.scores_mat is not None and self.categories is not None:
            cat_id = find_similar_category_id(language_desc, self.categories)
            scores_mat = self.scores_mat
            return scores_mat[:, cat_id].flatten()

        else:
            if with_init_cat:
                raise Exception(
                    "Categories are not preloaded. Call init_categories(categories: List[str]) to initialize categories."
                )

            text_feats = get_text_feats([language_desc], self.clip_model, self.clip_feat_dim)
            pose_scores = self.clip_sparse_map @ text_feats.T
            return pose_scores.flatten()
