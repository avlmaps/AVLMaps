import os
import subprocess
from collections import defaultdict
from pathlib import Path
import shutil
import gdown
from typing import Dict, List, Union

import habitat_sim
import hydra
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="generate_dataset.yaml",
)
def main(config: DictConfig) -> None:
    dataset_dir = Path(config.data_paths.avlmaps_data_dir) / "avlmaps_dataset"
    data_dirs = sorted([x for x in dataset_dir.iterdir() if x.is_dir()])
    if config.scene_names:
        data_dirs = sorted([dataset_dir / x for x in config.scene_names])
    levels = ["level_1", "level_2", "level_3"]
    stats = dict()
    overall_stats = defaultdict(int)
    pbar = tqdm(data_dirs)
    for data_dir_i, data_dir in enumerate(pbar):
        stats[data_dir.name] = defaultdict(int)
        pbar.set_description(desc=f"Scene {data_dir.name:14}")
        audio_video_dir = data_dir / "audio_video"
        seq_dirs = sorted([x for x in audio_video_dir.iterdir() if x.is_dir()])
        seq_pbar = tqdm(enumerate(seq_dirs), total=len(seq_dirs), leave=False)
        for seq_i, seq_dir in seq_pbar:
            seq_pbar.set_description(desc=f"Sequence {seq_dir.name:14}")
            for level in levels:
                video_path = seq_dir / f"output_with_audio_{level}.mp4"
                if video_path.exists():
                    stats[data_dir.name][level] += 1
                    overall_stats[level] += 1

    pbar = tqdm(data_dirs)
    for data_dir_i, data_dir in enumerate(pbar):
        print(data_dir.name)
        for level in levels:
            print(f"  {level}:", stats[data_dir.name][level])
    print("Overall")
    for level in levels:
        print(f"  {level}:", overall_stats[level])


if __name__ == "__main__":
    main()
