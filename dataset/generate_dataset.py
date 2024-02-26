import os
import subprocess
from pathlib import Path
import shutil
import gdown
from typing import Dict, List, Union

import habitat_sim
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from avlmaps.utils.habitat_utils import get_obj2cls_dict, make_cfg, save_obs
from avlmaps.utils.audio_utils import (
    generate_video_from_rgbs_batch,
    select_audio_frames_batch,
    assign_sound_to_video_batch,
)


def check_exist(save_id: int, root_save_dir: Path, sim_setting: Dict):
    flag = True
    if sim_setting["color_sensor"]:
        # save rgb
        save_name = f"{save_id:06}.png"
        save_path = root_save_dir / "rgb" / save_name
        flag = flag and save_path.exists()

    if sim_setting["depth_sensor"]:
        save_name = f"{save_id:06}.npy"
        save_path = root_save_dir / "depth" / save_name
        flag = flag and save_path.exists()

    if sim_setting["semantic_sensor"]:
        save_name = f"{save_id:06}.npy"
        save_path = root_save_dir / "semantic" / save_name
        flag = flag and save_path.exists()
    return flag


def generate_scene_data(
    save_dir: Union[Path, str],
    config: DictConfig,
    scene_path: Path,
    poses: np.ndarray,
    sim: habitat_sim.Simulator = None,
    reuse_sim: bool = False,
) -> None:
    """
    config: config for the sensors of the collected data
    scene_path: path to the Matterport3D scene file *.glb
    poses: (N, 7), each line has (px, py, pz, qx, qy, qz, qw)
    """
    sim_setting = {
        "scene": str(scene_path),
        "default_agent": 0,
        "sensor_height": config.camera_height,
        "color_sensor": config.rgb,
        "depth_sensor": config.depth,
        "semantic_sensor": config.semantic,
        "move_forward": 0.1,
        "turn_left": 5,
        "turn_right": 5,
        "width": config.resolution.w,
        "height": config.resolution.h,
        "enable_physics": False,
        "seed": 42,
    }
    if check_exist(len(poses) - 1, save_dir, sim_setting):
        return sim
    if sim is None:
        cfg = make_cfg(sim_setting)
        sim = habitat_sim.Simulator(cfg)

    # get the dict mapping object id to semantic id in this scene
    obj2cls = get_obj2cls_dict(sim)

    # initialize the agent in sim
    agent = sim.initialize_agent(sim_setting["default_agent"])
    pbar = tqdm(poses, leave=False)
    for pose_i, pose in enumerate(pbar):
        pbar.set_description(desc=f"Frame {pose_i:06}")
        if check_exist(pose_i, save_dir, sim_setting):
            continue
        agent_state = habitat_sim.AgentState()
        agent_state.position = pose[:3]
        agent_state.rotation = pose[3:]
        sim.get_agent(0).set_state(agent_state)
        obs = sim.get_sensor_observations(0)
        save_obs(save_dir, sim_setting, obs, pose_i, obj2cls)

    if not reuse_sim:
        sim.close()
        return None
    return sim


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="generate_dataset.yaml",
)
def main(config: DictConfig) -> None:
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.makedirs(config.data_paths.avlmaps_data_dir, exist_ok=True)
    dataset_dir = Path(config.data_paths.avlmaps_data_dir) / "avlmaps_dataset"
    if not dataset_dir.exists():
        zip_filepath = dataset_dir.parent / "avlmaps_dataset.zip"
        gdown.download(
            "https://drive.google.com/file/d/1Vrxpk2XFnk2WbJrnLNCyGgJ141JPoDDj/view?usp=sharing",
            zip_filepath.as_posix(),
            fuzzy=True,
        )
        subprocess.run(["unzip", zip_filepath.as_posix(), "-d", dataset_dir.parent.as_posix()])

    data_dirs = sorted([x for x in dataset_dir.iterdir() if x.is_dir()])
    if config.scene_names:
        data_dirs = sorted([dataset_dir / x for x in config.scene_names])
    audio_video_config = config.copy()
    audio_video_config.data_cfg.depth = False
    audio_video_config.data_cfg.semantic = False
    audio_video_config.data_cfg.resolution.w = 640
    audio_video_config.data_cfg.resolution.h = 480
    print(data_dirs)
    pbar = tqdm(data_dirs)
    for data_dir_i, data_dir in enumerate(pbar):
        pbar.set_description(desc=f"Scene {data_dir.name:14}")
        scene_name = data_dir.name.split("_")[0]
        scene_path = Path(config.data_paths.habitat_scene_dir) / scene_name / (scene_name + ".glb")
        pose_path = data_dir / "poses.txt"
        poses = np.loadtxt(pose_path)  # (N, 7), each line has (px, py, pz, qx, qy, qz, qw)
        generate_scene_data(data_dir, config.data_cfg, scene_path, poses)

        audio_video_dir = data_dir / "audio_video"
        seq_dirs = sorted([x for x in audio_video_dir.iterdir() if x.is_dir()])
        seq_pbar = tqdm(enumerate(seq_dirs), total=len(seq_dirs), leave=False)
        for seq_i, seq_dir in seq_pbar:
            seq_pbar.set_description(desc=f"Sequence {seq_dir.name:14}")
            pose_path = seq_dir / "poses.txt"
            poses = np.loadtxt(pose_path)
            if seq_i == 0:
                sim = generate_scene_data(seq_dir, audio_video_config.data_cfg, scene_path, poses, reuse_sim=True)
            else:
                sim = generate_scene_data(
                    seq_dir, audio_video_config.data_cfg, scene_path, poses, sim=sim, reuse_sim=True
                )

        if sim is not None:
            sim.close()

    print("Generating audio video for the dataset")
    fps = config["sound_data_collect_params"]["fps"]
    pbar = tqdm(data_dirs)
    for data_dir_i, data_dir in enumerate(pbar):

        print(f"Generating audio video for scene {data_dir_i}: {data_dir}")
        generate_video_from_rgbs_batch(data_dir, fps)
        select_audio_frames_batch(
            data_dir,
            interval_s=config["sound_data_collect_params"]["sound_interval"],
            min_dist_between_sound=config["sound_data_collect_params"]["min_dist_between_sound"],
            fps=fps,
            overwrite=config["sound_data_collect_params"]["overwrite_audio_frame_ranges"],
        )
        difficulties = ["level_1", "level_2", "level_3"]
        for difficulty in difficulties:
            np.random.seed(data_dir_i + config["sound_data_collect_params"]["seed"])
            assign_sound_to_video_batch(
                config["data_paths"]["esc50_audio_dir"],
                config["data_paths"]["esc50_meta_path"],
                data_dir,
                difficulty,
                config["sound_config"],
                fps,
                seq_num=config["sound_data_collect_params"]["considered_seq_num_per_scene"],
            )


if __name__ == "__main__":
    main()
