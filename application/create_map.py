from pathlib import Path
import hydra
from omegaconf import DictConfig
from avlmaps.map.avlmap import AVLMap


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_creation_cfg.yaml",
)
def main(config: DictConfig) -> None:
    avlmap = AVLMap(config)
    data_dir = Path(config.data_paths.avlmaps_data_dir) / "avlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    avlmap.create_map(data_dirs[config.scene_id])


if __name__ == "__main__":
    main()
