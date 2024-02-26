from pathlib import Path
import os
import json
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
import cv2
from omegaconf import DictConfig
import hydra
from avlmaps.map.interactive_map import InteractiveMap
from avlmaps.dataloader.habitat_dataloader import VLMapsDataloaderHabitat
from avlmaps.robot.habitat_lang_robot import HabitatLanguageRobot
from avlmaps.utils.mapping_utils import get_new_pallete
from avlmaps.utils.mapping_utils import cvt_pose_vec2tf

from typing import List, Dict, Tuple, Set


def load_meta(meta_path: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    categories = []
    frame_ranges = []
    with open(meta_path, "r") as f:
        for line in f:
            row = line.split(",")
            frame_range = (int(row[0]), int(row[1]))
            category = row[4]
            frame_ranges.append(frame_range)
            categories.append(category)
    return categories, frame_ranges


def load_audio_video_gt_cat_and_poses(
    data_dir: str,
    seq_dir: str,
    vlmaps_dataloader: VLMapsDataloaderHabitat,
    difficulty_level: str,
) -> Tuple[List[str], List[List[Tuple[float, float, float]]]]:
    # load range_and_audio_meta.txt
    # get lists of audio categories and frame ranges
    meta_path = os.path.join(seq_dir, f"range_and_audio_meta_{difficulty_level}.txt")
    categories, frame_ranges = load_meta(meta_path)

    # get the list of poses associated with each category
    # pose_dir = os.path.join(seq_dir, "pose")
    # pose_paths_list = [os.path.join(pose_dir, x) for x in sorted(os.listdir(pose_dir))]
    pose_path = Path(seq_dir) / "poses.txt"
    pose_list = np.loadtxt(pose_path)

    # convert poses to map positions (make use of vlmaps_dataloader)
    # vlmaps_dataloader = VLMapsDataloaderHabitat(data_dir, "gt")
    map_poses = []
    for iter_i, (cat, frame_range) in enumerate(zip(categories, frame_ranges)):
        # pose_paths = pose_paths_list[frame_range[0] : frame_range[1] + 1]
        poses_list_in_range = pose_list[frame_range[0] : frame_range[1] + 1]
        cat_map_poses = []
        for pose_i, posevec in enumerate(poses_list_in_range):
            tf_hab = cvt_pose_vec2tf(posevec)

            # pos, rot = load_pose(pose_path)
            # tf_hab = np.eye(4)
            # tf_hab[:3, :3] = rot
            # tf_hab[:3, 3] = pos.flatten()
            vlmaps_dataloader.from_habitat_tf(tf_hab)
            (
                row,
                col,
                angle_deg,
            ) = vlmaps_dataloader.to_cropped_map_pose()
            cat_map_poses.append((row, col, angle_deg))

        map_poses.append(cat_map_poses)
    return categories, map_poses


def draw_audio_video_gt_on_map(
    map: np.ndarray,
    categories: List[str],
    map_pos: List[List[Tuple[float, float]]],
    radius_pix: float,
    zoom_times: int = 2,
) -> np.ndarray:
    masked_map = map.copy()
    h, w, _ = masked_map.shape
    masked_map = cv2.resize(masked_map, (int(w * zoom_times), int(h * zoom_times)))

    new_pallete = get_new_pallete(len(categories))
    new_pallete = np.array(new_pallete).reshape((-1, 3)).astype(np.uint8)
    for iter_i, (cat, pos_list, color) in enumerate(zip(categories, map_pos, new_pallete)):
        color = (int(color[0]), int(color[1]), int(color[2]))
        for pose_i, pos in enumerate(pos_list):
            cv2.circle(
                masked_map,
                (pos[1] * zoom_times, pos[0] * zoom_times),
                radius_pix,
                color,
                2,
            )

    text_positions = [np.array([10000, 10000])]
    for iter_i, (cat, pos_list, color) in enumerate(zip(categories, map_pos, new_pallete)):
        print(f"inserting category {iter_i} {cat}")
        color = (int(color[0]), int(color[1]), int(color[2]))
        inserted = False
        for pos_i, pos in enumerate(pos_list):
            pos = np.array([pos[0] * zoom_times - 10, pos[1] * zoom_times])
            old_positions = np.array(text_positions).reshape((-1, 2))
            dists = np.linalg.norm(old_positions - pos, axis=1)
            min_dist = np.min(dists)
            min_row_dist = np.min((old_positions - pos)[:, 0])
            if min_dist > 10 and min_row_dist > 10:
                cv2.putText(
                    masked_map,
                    f"{cat}",
                    (pos[1], pos[0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )
                text_positions.append(pos)
                inserted = True
                break

        if inserted == False:
            print("fail to find position")
            cv2.putText(
                masked_map,
                f"{cat}",
                (
                    pos_list[len(pos_list) // 2][1] * zoom_times,
                    pos_list[len(pos_list) // 2][0] * zoom_times,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

    return masked_map


def draw_poses_on_map(
    map: np.ndarray,
    poses_list: List[Tuple[float, float, float]],
    color: Tuple[int, int, int] = (0, 0, 255),
    ignore_diretion: bool = False,
) -> np.ndarray:
    for pose_i, pose in enumerate(poses_list):
        if not ignore_diretion:
            row, col, theta = pose[0], pose[1], np.deg2rad(pose[2])
            ep = [int(col + 20 * np.sin(theta)), int(row - 20 * np.cos(theta))]
            map = cv2.line(map, (col, row), ep, color)
        cv2.circle(map, (pose[1], pose[0]), 3, color, 1)
    return map


def draw_goal_pos_on_map(map: np.ndarray, goals_list: List[Tuple[float, float, float]]) -> np.ndarray:
    mask = map.copy()
    for pos_list_i, pos_list in enumerate(goals_list):
        for pos_i, pos in enumerate(pos_list):
            cv2.circle(mask, (pos[1], pos[0]), 3, (0, 0, 255), 1)
    return mask


def auto_generate_nav_start_poses_on_map(
    vlmaps_dataloader: VLMapsDataloaderHabitat,
    all_map_pos_cropped: List[List[Tuple[float, float]]],
    poses_num: int = 1000,
    radius_pix: float = 20,
) -> List[Tuple[float, float, float]]:
    """
    Generate a list of full map poses
    """
    floor_mask = vlmaps_dataloader.gt_cropped == 2
    obst_cropped = vlmaps_dataloader.obstacles_cropped
    obst_cropped[floor_mask] = 1
    obst_cropped = obst_cropped == 0
    obst_cropped = Map._dilate_map(obst_cropped, dilate_iter=6)
    free_cropped = obst_cropped == 0
    mask = np.zeros_like(vlmaps_dataloader.obstacles_cropped, dtype=np.uint8)
    for list_i, pos_list in enumerate(all_map_pos_cropped):
        for pos_i, pos in enumerate(pos_list):
            cv2.circle(mask, (int(pos[1]), int(pos[0])), radius_pix, 255, -1)
    mask = np.logical_and(mask, free_cropped)
    rows, cols = np.where(mask > 0)
    ids = np.random.choice(rows.shape[0], poses_num)
    selected_degs = np.random.rand(poses_num) * 360 - 180
    selected_rows = rows[ids] + vlmaps_dataloader.xmin
    selected_cols = cols[ids] + vlmaps_dataloader.ymin
    return [(row, col, deg) for (row, col, deg) in zip(selected_rows, selected_cols, selected_degs)]


def filter_goals_with_distance(
    start_pos: Tuple[float, float],
    goal_positions: List[List[Tuple[float, float]]],
    dist_range: Tuple[float, float],
) -> List[int]:
    start_pos = np.array(start_pos)
    goal_pos = [np.stack(pos_list, axis=0) for pos_list in goal_positions]

    min_dists = np.array([np.min(np.linalg.norm(goals - start_pos, axis=-1)) for goals in goal_pos])
    ids = np.where(np.logical_and(min_dists > dist_range[0], min_dists < dist_range[1]))[0]

    return ids.astype(int)


def find_goals_within_range(
    start_pos: Tuple[float, float],
    goal_positions: List[List[Tuple[float, float]]],
    max_dist: float,
) -> List[int]:
    start_pos = np.array(start_pos)
    goal_pos = [np.stack(pos_list, axis=0) for pos_list in goal_positions]

    min_dists = np.array([np.min(np.linalg.norm(goals - start_pos, axis=1)) for goals in goal_pos])
    ids = np.where(min_dists < max_dist)[0]

    return ids.astype(int)


def find_nearest_pos(pos: Tuple[float, float], pos_list: List[Tuple[float, float]]) -> int:
    pos = np.array(pos)
    pos_list = np.array(pos_list)
    dists = np.linalg.norm(pos_list - pos, axis=-1)
    id = np.argmin(dists)
    return id


def find_goal_in_range(
    curr_pos: Tuple[float, float],
    candidate_positions: List[List[Tuple[float, float]]],
    dist_range_pix: Tuple[float, float],
    assigned_ids: Set[int],
):
    ids = []
    dist_range = dist_range_pix
    while len(ids) == 0:
        ids = filter_goals_with_distance(curr_pos, candidate_positions, dist_range)
        ids = [x for x in ids if x not in assigned_ids]
        # near_ids = find_goals_within_range(curr_pos, candidate_positions, dist_range[0])
        # print(ids)
        # print(near_ids)
        # removed_cat_set = {categories[i] for i in near_ids}
        # print(removed_cat_set)
        # ids = [id for id in ids if categories[id] not in removed_cat_set]
        # print("ids: ", ids)
        dist_range = (dist_range[0], dist_range[1] + 10)
        # print(dist_range)
        if dist_range[1] > 1000:
            break
    return ids


def select_sound_goals(
    start_pos_cropped: Tuple[float, float],
    categories: List[str],
    map_pos_cropped: List[List[Tuple[float, float]]],
    goals_num: int = 4,
    distance_range_pix: Tuple[float, float] = (40, 100),
) -> List[str]:

    goal_ids = []
    curr_pos = start_pos_cropped

    assigned_ids = set()
    for i in range(goals_num):
        # filter out near goals
        ids = find_goal_in_range(curr_pos, map_pos_cropped, distance_range_pix, assigned_ids)
        if len(ids) == 0:
            return []

        print(ids)
        picked_id = np.random.choice(ids)
        print("picked id: ", picked_id)
        goal_ids.append(picked_id)
        assigned_ids.add(picked_id)
        nearest_id = find_nearest_pos(curr_pos, map_pos_cropped[picked_id])
        curr_pos = map_pos_cropped[picked_id][nearest_id]
    return goal_ids


def save_sound_nav_tasks(
    save_dir: str,
    difficulty_level: str,
    init_hab_poses: List[np.ndarray],
    categories: List[List[str]],
    goal_positions: List[Tuple[float, float]],
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"sound_nav_tasks_{difficulty_level}.txt")
    with open(save_path, "w") as f:
        for task_i, (init_tf, goal_categories) in enumerate(zip(init_hab_poses, categories)):
            xyz_quat_xyzw_str = ",".join([str(x) for x in init_tf.flatten()])
            goal_categories_str = ",".join(goal_categories)
            f.write(f"{xyz_quat_xyzw_str},{goal_categories_str}\n")


def save_cross_modality_nav_tasks(
    save_dir: str,
    task_id: int,
    difficulty_level: str,
    init_hab_tf: np.ndarray,
    instruction: str,
    goal_positions_full: List[List[str]],
):
    save_path = os.path.join(save_dir, f"{task_id:06}_{difficulty_level}.txt")
    init_tf_str = ",".join([str(x) for x in init_hab_tf.flatten().tolist()])
    with open(save_path, "w") as f:
        f.write(init_tf_str + "\n")
        f.write(instruction + "\n")
        separator = ""
        for group_i, group_positions in enumerate(goal_positions_full):
            pos_str = separator
            separator = "\n"
            sep = ""
            for pos_i, pos in enumerate(group_positions):
                pos_str += sep + str(pos[0]) + "," + str(pos[1])
                sep = ";"

            f.write(pos_str)
