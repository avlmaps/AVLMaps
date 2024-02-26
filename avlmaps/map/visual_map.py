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

from avlmaps.utils.mapping_utils import get_sim_cam_mat_with_fov
from avlmaps.utils.localization_utils import HLocLocalizer


class VisualMap:
    def __init__(self, map_config: DictConfig, data_dir: str = "") -> None:
        self.map_config = map_config
        self.pose_info = map_config.pose_info
        self.ref_cam_intrinsic_mat = np.array(self.map_config.cam_calib_mat).reshape((3, 3))
        self.tf_base2cam = np.eye(4)
        self.tf_base2cam[:3, :3] = np.array(self.pose_info.base2cam_rot).reshape((3, 3))
        self.tf_base2cam[1, 3] = self.pose_info.camera_height
        if data_dir:
            self._setup_paths(data_dir)

    def _setup_localizer(self, data_dir: str):
        data_dir = Path(data_dir)
        self._setup_paths(data_dir)
        self.localizer = HLocLocalizer(self.map_save_dir)
        self.localizer.init_video_with_images_folder(self.rgb_dir, frame_sample_interval=1, key=None)
        self.localizer.init_depth_with_depth_folder(self.depth_dir, frame_sample_interval=1, key=None)
        self.localizer.init_pose_with_pose_file(self.pose_path, frame_sample_interval=1)
        self.localizer.compute_global_descriptor(
            self.localizer.image_paths_list,
            reference=True,
            overwrite=False,
            descriptor_filename=data_dir.stem,
        )

    def _setup_paths(self, data_dir: Union[Path, str]) -> None:
        self.data_dir = Path(data_dir)
        self.rgb_dir = self.data_dir / "rgb"
        self.depth_dir = self.data_dir / "depth"
        self.pose_path = self.data_dir / "poses.txt"
        self.map_save_dir = self.data_dir / "visual_map"
        os.makedirs(self.map_save_dir, exist_ok=True)
        try:
            self.rgb_paths = sorted(self.rgb_dir.glob("*.png"))
            self.depth_paths = sorted(self.depth_dir.glob("*.npy"))
        except FileNotFoundError as e:
            print(e)

    def create_and_load_map(self, data_dir: Union[Path, str]) -> None:
        self._setup_localizer(data_dir)

    def localize_image(
        self,
        img: np.ndarray,
        query_cam_intrinsic_mat: np.ndarray = None,
        sim_cam_fov: float = 90,
        vis: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if query_cam_intrinsic_mat is None:
            query_cam_intrinsic_mat = get_sim_cam_mat_with_fov(img.shape[0], img.shape[1], sim_cam_fov)

        init_tf_cam = self.localizer.pose_list[0] @ self.tf_base2cam
        init_tf_cam_inv = np.linalg.inv(init_tf_cam)
        ref_img_id, transform = self.localizer.localize_agent_with_depth(
            img, ref_intr_mat=self.ref_cam_intrinsic_mat, query_intr_mat=query_cam_intrinsic_mat, vis=vis
        )
        if ref_img_id == -1:
            return None

        tf = self.localizer.pose_list[ref_img_id] @ self.tf_base2cam
        query_cam_tf = tf @ transform
        query_base_tf = query_cam_tf @ np.linalg.inv(self.tf_base2cam)
        return query_cam_tf, query_base_tf
