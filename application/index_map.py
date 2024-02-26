import os
from pathlib import Path
import hydra
from omegaconf import DictConfig
import numpy as np
import cv2
import habitat_sim

from avlmaps.map.avlmap import AVLMap
from avlmaps.map.interactive_map import InteractiveMap
from avlmaps.utils.habitat_utils import make_cfg
from avlmaps.utils.mapping_utils import get_sim_cam_mat_with_fov
from avlmaps.utils.visualize_utils import visualize_rgb_map_3d, visualize_heatmap_3d
from avlmaps.utils.sound_nav_collect_utils import load_audio_video_gt_cat_and_poses, draw_audio_video_gt_on_map
from avlmaps.utils.audio_utils import setup_audio_paths


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="map_indexing_cfg.yaml",
)
def main(config: DictConfig) -> None:
    data_dir = Path(config.data_paths.avlmaps_data_dir) / "avlmaps_dataset"
    data_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])
    avlmap = AVLMap(config, data_dir=data_dirs[config.scene_id])
    avlmap.load_map(data_dirs[config.scene_id])
    avlmap.vlmap._init_clip()
    interactive_map = InteractiveMap(data_dirs[config.scene_id], config.map_config)
    sim = None

    while True:
        index_type = input(
            "What do you want to index? (1. object, 2. sound, 3. area, 4. image, 5. show rgb point cloud, or 6. exit)\nInput: "
        )
        if index_type == "1":
            obj_name = input("What is the object name you want to index?\nInput: ")
            heatmap = avlmap.index_object(obj_name, decay_rate=0.01)
        elif index_type == "2":
            color_top_down = interactive_map.vlmaps_dataloader.map.generate_rgb_topdown_map() / 255.0
            color_top_down = color_top_down[
                interactive_map.vlmaps_dataloader.rmin : interactive_map.vlmaps_dataloader.rmax + 1,
                interactive_map.vlmaps_dataloader.cmin : interactive_map.vlmaps_dataloader.cmax + 1,
            ]

            # # load scene GT sound positions as a reference
            level = config.sound_data_collect_params.difficulty
            audio_video_dir, seq_dirs = setup_audio_paths(data_dirs[config.scene_id])
            gt_categories = []
            gt_map_poses = []
            for seq_dir in seq_dirs:
                categories, map_poses = load_audio_video_gt_cat_and_poses(
                    data_dirs[config.scene_id],
                    seq_dir,
                    vlmaps_dataloader=interactive_map.vlmaps_dataloader,
                    difficulty_level=level,
                )
                gt_categories.extend(categories)
                gt_map_poses.extend(map_poses)

            color_top_down = cv2.cvtColor((color_top_down * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            masked_map = draw_audio_video_gt_on_map(color_top_down, gt_categories, gt_map_poses, 3)
            cv2.imshow("GT sound positions", masked_map)
            cv2.waitKey()

            sound_name = input("What is the sound name you want to index?\nInput: ")
            heatmap = avlmap.index_sound(sound_name, decay_rate=0.01)
        elif index_type == "3":
            area_name = input("What is the area name you want to index?\nInput: ")
            heatmap = avlmap.index_area(area_name, decay_rate=0.01)
        elif index_type == "4":
            color_top_down = interactive_map.vlmaps_dataloader.map.generate_rgb_topdown_map() / 255.0
            color_top_down = color_top_down[
                interactive_map.vlmaps_dataloader.rmin : interactive_map.vlmaps_dataloader.rmax + 1,
                interactive_map.vlmaps_dataloader.cmin : interactive_map.vlmaps_dataloader.cmax + 1,
            ]

            interactive_map.collect_map_positions(color_top_down)
            tf_hab, agent_state = interactive_map.get_habitat_robot_state(
                interactive_map.coords[0], interactive_map.coords[1]
            )

            scene_name = data_dirs[config.scene_id].name.split("_")[0]
            scene_path = Path(config.data_paths.habitat_scene_dir) / scene_name / (scene_name + ".glb")
            sim_setting = {
                "scene": str(scene_path),
                "default_agent": 0,
                "sensor_height": config.image_query_cfg.camera_height,
                "color_sensor": True,
                "depth_sensor": False,
                "semantic_sensor": False,
                "move_forward": 0.1,
                "turn_left": 5,
                "turn_right": 5,
                "width": config.image_query_cfg.resolution.w,
                "height": config.image_query_cfg.resolution.h,
                "enable_physics": False,
                "seed": 42,
            }
            os.environ["MAGNUM_LOG"] = "quiet"
            os.environ["HABITAT_SIM_LOG"] = "quiet"
            if sim is None:
                print("Initializing simulator")
                cfg = make_cfg(sim_setting)
                sim = habitat_sim.Simulator(cfg)

            # visualize the query image
            sim.get_agent(0).set_state(agent_state)
            observations = sim.get_sensor_observations()
            rgb = observations["color_sensor"][:, :, :3]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow("query image", bgr)

            # show the pose in top down map
            src = (int(interactive_map.coords[0][0]), int(interactive_map.coords[0][1]))
            tar = (int(interactive_map.coords[1][0]), int(interactive_map.coords[1][1]))
            color_top_down = cv2.cvtColor((color_top_down * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.arrowedLine(
                color_top_down,
                src,
                tar,
                (0, 0, 255),
                2,
            )
            cv2.imshow("query pose top down", color_top_down)
            cv2.waitKey()

            # save query image
            if config.image_query_cfg.save_query_image:
                query_image_dir = Path(__file__).parent / "query_images"
                os.makedirs(query_image_dir, exist_ok=True)
                save_path = query_image_dir / "query_image.png"
                print("query image saved at: ", save_path)
                cv2.imwrite(str(save_path), bgr)
                save_path = query_image_dir / "query_image_top_down.png"
                print("query image saved at: ", save_path)
                cv2.imwrite(str(save_path), color_top_down)

            query_cam_intrinsic_mat = get_sim_cam_mat_with_fov(
                config.image_query_cfg.resolution.h, config.image_query_cfg.resolution.w, fov=90
            )
            heatmap = avlmap.index_image(rgb, query_cam_intrinsics=query_cam_intrinsic_mat)
            # visualize_rgb_map_3d(avlmap.vlmap.grid_pos, avlmap.vlmap.grid_rgb)
        elif index_type == "5":
            visualize_rgb_map_3d(avlmap.vlmap.grid_pos, avlmap.vlmap.grid_rgb)
            continue
        else:
            break
        visualize_heatmap_3d(avlmap.vlmap.grid_pos, heatmap, avlmap.vlmap.grid_rgb)


if __name__ == "__main__":
    main()
