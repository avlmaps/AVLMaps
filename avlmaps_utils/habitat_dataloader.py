import os
import numpy as np

from avlmaps_utils.clip_mapping_utils import *
from avlmaps_utils.clip_utils import *
from avlmaps_utils.time_utils import Tic

from typing import Tuple, List, Dict, Optional


class VLMapsDataloaderHabitat:
    def __init__(
        self,
        data_dir: str,
        cs: float = 0.05,
        gs: int = 1000,
        camera_height: float = 1.5,
        tf_ro_cam: Optional[np.ndarray] = None
    ):
        self.data_dir = data_dir
        is_lanseg = True

        map_save_dir = os.path.join(data_dir, "map")

        os.makedirs(map_save_dir, exist_ok=True)

        self.obstacles_save_path = os.path.join(map_save_dir, "obstacles.npy")

        # initialize a grid with zero at the center
        self.cs = cs
        self.gs = gs

        tic = Tic()
        self.obstacles = load_map(self.obstacles_save_path)
        x_indices, y_indices = np.where(self.obstacles == 0)
        tic.print_time("loading obstacles map")




        self.xmin = np.min(x_indices)
        self.xmax = np.max(x_indices)
        self.ymin = np.min(y_indices)
        self.ymax = np.max(y_indices)

        self.obstacles_cropped = self.obstacles[
            self.xmin : self.xmax + 1, self.ymin : self.ymax + 1
        ]

        self.gt_semantic_map_save_path = os.path.join(
            map_save_dir, f"grid_gt_1.npy"
        )
        tic.tic()
        self.gt = load_map(self.gt_semantic_map_save_path)
        self.gt_cropped = self.gt[self.xmin : self.xmax + 1, self.ymin : self.ymax + 1]
        tic.print_time("loading gt")

        self.no_map_mask = self.obstacles_cropped > 0

        pose_dir = os.path.join(data_dir, "pose")
        self.pose_list = sorted(
            os.listdir(pose_dir), key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        self.pose_list = [os.path.join(pose_dir, x) for x in self.pose_list]

        # load mapping init habitat pose
        init_pos, init_rot = load_pose(self.pose_list[0])
        self.map_init_pose = np.eye(4)
        self.map_init_pose[:3, :3] = init_rot
        self.map_init_pose[:3, 3] = init_pos.reshape(-1)

        self.camera_height = camera_height
        self.tf_ro_cam = tf_ro_cam
        if self.tf_ro_cam is None:
            self.tf_ro_cam = np.eye(4)
            self.tf_ro_cam[1, 1] = -1
            self.tf_ro_cam[2, 2] = -1
            self.tf_ro_cam[1, 3] = camera_height
    

    def get_obstacles_cropped(self) -> np.array:
        return self.obstacles_cropped

    def get_obstacles_cropped_no_floor(self) -> np.array:
        floor_mask = self.gt_cropped == 2
        obstacles_cropped_no_floor = self.obstacles_cropped.copy()
        obstacles_cropped_no_floor[floor_mask] = 1
        return obstacles_cropped_no_floor

    def convert_cropped_map_point_to_camera_position(
        self, col: int, row: int
    ) -> Tuple[float, float]:
        col_full = self.ymin + col
        row_full = self.xmin + row
        col_hab, row_hab = grid_id2pos(self.gs, self.cs, col_full, row_full)
        return col_hab, row_hab

    def compute_map_orientation(
        self, src_col: int, src_row: int, tar_col: int, tar_row: int
    ) -> float:
        theta = np.arctan2(tar_col - src_col, -tar_row + src_row)
        return theta

    def convert_cropped_map_pose_to_habitat_pose(
        self, row: int, col: int, theta_deg: float
    ) -> np.ndarray:
        x, z = self.convert_cropped_map_point_to_camera_position(col, row)
        theta = np.deg2rad(theta_deg)

        tf = np.eye(4)
        tf[:3, 3] = [x, 0, z]
        tf[0, 0] = np.cos(theta)
        tf[2, 2] = np.cos(theta)
        tf[0, 2] = np.sin(theta)
        tf[2, 0] = -np.sin(theta)

        tf_hab = (
            self.map_init_pose @ self.tf_ro_cam @ tf @ np.linalg.inv(self.tf_ro_cam)
        )

        return tf_hab

    def convert_full_map_pose_to_habitat_pose(
        self, row: int, col: int, theta_deg: float
    ) -> np.ndarray:
        row_cropped, col_cropped = self.convert_full_map_pos_to_cropped_map_pos((row, col))
        return self.convert_cropped_map_pose_to_habitat_pose(row_cropped, col_cropped, theta_deg)

    def convert_habitat_tf_to_cam_tf(self, tf_hab: np.ndarray) -> np.ndarray:
        tf_cam = tf_hab @ self.tf_ro_cam
        return tf_cam


    def convert_habitat_tf_to_full_map_pose(self, tf_hab: np.ndarray) -> Tuple[float, float, float]:
        tf_cam = np.linalg.inv(self.map_init_pose @ self.tf_ro_cam) @ tf_hab @ self.tf_ro_cam
        x, z = tf_cam[[0, 2], 3]
        theta = np.arctan2(tf_cam[0, 2], tf_cam[2, 2])
        theta_deg = np.rad2deg(theta)
        col, row = pos2grid_id(self.gs, self.cs, x, z)
        return (row, col, theta_deg)
    
    def convert_camera_tf_to_habitat_tf(self, tf_cam: np.ndarray) -> np.ndarray:
        return self.map_init_pose @ self.tf_ro_cam @ tf_cam @ np.linalg.inv(self.tf_ro_cam)

    def convert_habitat_tf_to_cropped_map_pose(self, tf_hab: np.ndarray) -> Tuple[float, float, float]:
        (row, col, theta_deg) = self.convert_habitat_tf_to_full_map_pose(tf_hab)
        row, col = self.convert_full_map_pos_to_cropped_map_pos((row, col))
        return (row, col, theta_deg)
    
    def convert_habitat_pos_list_to_cropped_map_pos_list(self, pos_list: List[np.ndarray]) -> List[Tuple[float, float]]:
        cropped_pos_list = []
        for pos_i, pos in enumerate(pos_list):
            tf = np.eye(4)
            tf[:3, 3] = pos.flatten()
            (row, col, angle_deg) = self.convert_habitat_tf_to_cropped_map_pose(tf)
            cropped_pos_list.append((row, col))
        return cropped_pos_list
    
    def convert_habitat_pos_list_to_full_map_pos_list(self, pos_list: List[np.ndarray]) -> List[Tuple[float, float]]:
        full_pos_list = []
        for pos_i, pos in enumerate(pos_list):
            tf = np.eye(4)
            tf[:3, 3] = pos.flatten()
            (row, col, angle_deg) = self.convert_habitat_tf_to_full_map_pose(tf)
            full_pos_list.append((row, col))
        return full_pos_list

    def convert_full_map_pos_to_cropped_map_pos(self, full_map_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        full_map_pos: (row, col) in full map
        Return (row, col) in cropped_map
        """
        return [full_map_pos[0] - self.xmin, full_map_pos[1] - self.ymin]

    def convert_cropped_map_pos_to_full_map_pos(self, cropped_map_pos: Tuple[float, float]) -> Tuple[float, float]:
        """
        cropped_map_pos: (row, col) in cropped map
        Return (row, col) in full map
        """
        return [cropped_map_pos[0] + self.xmin, cropped_map_pos[1] + self.ymin]
