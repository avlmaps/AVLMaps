import os
import hydra
from omegaconf import DictConfig
import numpy as np
import cv2
import torch
import clip
import open3d as o3d

from matplotlib import pyplot as plt

from avlmaps_utils.clip_mapping_utils import *
from avlmaps_utils.clip_utils import get_text_feats
# from utils.planning_utils import mp3dcat
from avlmaps_utils.time_utils import Tic
from avlmaps_utils.mapping_3d_utils import generate_fused_features, load_3d_map

def visualize_vlmaps_3d(data_dir: str, camera_height: float, cs: float, gs: int):
    # loading models
    tic = Tic()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # self.clip_version = "ViT-L/14"
    clip_version = "ViT-B/32"
    clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                    'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
    clip_model.to(device).eval()
    tic.print_time("load clip model")

    tic.tic()
    lang = ["other", "paper box", "backpack", "chair", "counter", "table", "shelf", "door", "microwave oven", "bottle", "cabinet"]
    lang = ["a picture of " + x for x in lang]
    text_feat = get_text_feats(lang, clip_model, clip_feat_dim) # (N, 512)
    print("norm text feats: ", np.linalg.norm(text_feat, axis=1))
    print("text feats: ", text_feat.shape)
    tic.print_time("get text feats")

    # load concept fusion map
    tic.tic()
    map_save_dir = os.path.join(data_dir, "vlmaps_lseg_3d")
    concept_fusion_map_save_path = os.path.join(map_save_dir, f"vlmaps_lseg_3d.h5df")
    mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, _ = load_3d_map(concept_fusion_map_save_path)
    tic.print_time("load concept fusion map")
    pc_global = grid_id2pos_3d_batch(grid_pos, camera_height, cs, gs) # (N, 3)

    similarity = grid_feat @ text_feat.T # (N, 6)
    non_nan_mask = ~np.isnan(np.sum(similarity, axis=1))
    similarity = similarity[non_nan_mask, :]
    pc_global = pc_global[non_nan_mask, :]
    print(np.max(similarity), np.min(similarity))
    print(similarity.shape)
    print(similarity[:10, :])
    predict = np.argmax(similarity, axis=1)

    for i in range(similarity.shape[1]):
        print(lang[i])

        sim = predict == i
        sim = (sim * 200).astype(np.uint8)

        rgb_pc = cv2.applyColorMap(sim, cv2.COLORMAP_JET)
        rgb_pc = rgb_pc.reshape(-1, 3)[:, ::-1].astype(np.float32) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_global)
        pcd.colors = o3d.utility.Vector3dVector(rgb_pc)
        o3d.visualization.draw_geometries([pcd])


    pass

@hydra.main(
    version_base=None,
    config_path="../../config/habitat",
    # config_name="concept_fusion_config.yaml",
    config_name="vlmaps_lseg_real_world_config.yaml",
)
def main(config: DictConfig):
    data_root_dir = config.data_paths.vlmaps_data_dir
    data_dir_base = config.data_dir_base
    data_dir = os.path.join(data_root_dir, data_dir_base)
    camera_height = config.camera_height
    cs = config.cs
    gs = config.gs
    depth_sample_rate = config.depth_sample_rate
    visualize_vlmaps_3d(data_dir, camera_height, cs, gs)


if __name__ == '__main__':
    main()