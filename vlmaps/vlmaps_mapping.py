import time
import os
import tqdm

import hydra
from omegaconf import DictConfig
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import clip

from matplotlib import pyplot as plt

from avlmaps_utils.clip_mapping_utils import *
from avlmaps_utils.time_utils import Tic
from avlmaps_utils.concept_fusion_utils import (
    generate_fused_features,
    save_concept_fusion_map,
    load_concept_fusion_map,
)
from avlmaps_utils.lseg_utils import get_lseg_feat

from lseg.modules.models.lseg_net import LSegEncNet
from lseg.additional_utils.models import resize_image, pad_image, crop_image


def create_vlmaps_3d_batch(
    data_dir: str,
    camera_height: float,
    cs: float = 0.05,
    gs: int = 1000,
    depth_sample_rate: int = 100,
):
    print("data_dir: ", data_dir)

    crop_size = 480  # 480
    base_size = 520  # 520
    lang = "door,chair,ground,ceiling,other"
    labels = lang.split(",")
    vis = False

    # loading models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # self.clip_version = "ViT-L/14"
    clip_version = "ViT-B/32"
    clip_feat_dim = {
        "RN50": 1024,
        "RN101": 512,
        "RN50x4": 640,
        "RN50x16": 768,
        "RN50x64": 1024,
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
    }[clip_version]
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
    clip_model.to(device).eval()

    print("Creating LSegEncNet model...")
    st = time.time()
    model = LSegEncNet(
        lang, arch_option=0, block_depth=0, activation="lrelu", crop_size=crop_size
    )
    et = time.time()
    print(f"Creating time {et - st}s.")
    model_state_dict = model.state_dict()
    print("Loading pretrained model...")
    st = time.time()
    pretrained_state_dict = torch.load("lseg/checkpoints/demo_e200.ckpt")
    et = time.time()
    print(f"Loading time {et - st}s.")
    print("Filtering pretrained model...")
    pretrained_state_dict = {
        k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()
    }
    print("Assigning pretrained model parameters to model...")
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict)

    model.eval()
    model = model.cuda()

    norm_mean = [0.5, 0.5, 0.5]
    norm_std = [0.5, 0.5, 0.5]
    padding = [0.0] * 3
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            # transforms.Resize([360,480]),
        ]
    )

    print(f"loading scene {data_dir}")
    rgb_dir = os.path.join(data_dir, "rgb")
    depth_dir = os.path.join(data_dir, "depth")
    pose_dir = os.path.join(data_dir, "pose")

    rgb_list = sorted(
        os.listdir(rgb_dir), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    depth_list = sorted(
        os.listdir(depth_dir), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    pose_list = sorted(
        os.listdir(pose_dir), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    rgb_list = [os.path.join(rgb_dir, x) for x in rgb_list]
    depth_list = [os.path.join(depth_dir, x) for x in depth_list]
    pose_list = [os.path.join(pose_dir, x) for x in pose_list]

    map_save_dir = os.path.join(data_dir, "vlmaps_lseg_3d")
    os.makedirs(map_save_dir, exist_ok=True)

    concept_fusion_map_save_path = os.path.join(map_save_dir, f"vlmaps_lseg_3d.h5df")

    vh = int(camera_height / cs)
    grid_feat = np.zeros((gs * gs, clip_feat_dim), dtype=np.float32)
    grid_pos = np.zeros((gs * gs, 3), dtype=np.int32)
    occupied_ids = -1 * np.ones((gs, gs, vh), dtype=np.int32)
    weight = np.zeros((gs * gs), dtype=np.float32)
    grid_rgb = np.zeros((gs * gs, 3), dtype=np.uint8)

    max_id = 0
    mapped_iter_set = set()
    if os.path.exists(concept_fusion_map_save_path):
        (
            mapped_iter_list,
            grid_feat,
            grid_pos,
            weight,
            occupied_ids,
            grid_rgb,
        ) = load_concept_fusion_map(concept_fusion_map_save_path)
        mapped_iter_set = set(mapped_iter_list)
        print(grid_feat.shape)
        print(np.max(occupied_ids))
        max_id = grid_feat.shape[0]
    pos, rot = load_pose(pose_list[0])
    rot_ro_cam = np.eye(3)
    rot_ro_cam[1, 1] = -1
    rot_ro_cam[2, 2] = -1
    rot = rot @ rot_ro_cam
    pos[1] += camera_height

    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos.reshape(-1)
    init_tf_inv = np.linalg.inv(pose)

    tf_list = []
    # load all images and depths and poses
    data_iter = zip(rgb_list, depth_list, pose_list)
    pbar = tqdm(total=len(rgb_list))
    for iter_i, data_sample in enumerate(data_iter):
        rgb_path, depth_path, pose_path = data_sample
        if iter_i in mapped_iter_set:
            continue
        st = time.time()

        tic = Tic()
        bgr = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # read pose
        pos, rot = load_pose(pose_path)  # z backward, y upward, x to the right
        rot_ro_cam = np.eye(3)
        rot_ro_cam[1, 1] = -1
        rot_ro_cam[2, 2] = -1
        rot = rot @ rot_ro_cam
        pos[1] += camera_height

        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = pos.reshape(-1)

        tf_list.append(pose)

        tf = init_tf_inv @ pose
        theta = np.arctan2(tf[0, 2], tf[2, 2])

        # read depth
        depth = load_depth(depth_path)
        depth = depth / 10.0
        depth = depth * 10.0

        # pix_feats = generate_fused_features(rgb_path, bboxes_path, mask_path, clip_model, preprocess) # (H, W, 512)
        pix_feats = get_lseg_feat(
            model,
            rgb,
            labels,
            transform,
            crop_size,
            base_size,
            norm_mean,
            norm_std,
            vis,
        )

        # project all point cloud onto the ground, once there are points in a cell,
        pc, mask = depth2pc(depth)
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        pc_global = transform_pc(pc, tf)

        rgb_cam_mat = get_sim_cam_mat(rgb.shape[0], rgb.shape[1])
        feat_cam_mat = get_sim_cam_mat(pix_feats.shape[2], pix_feats.shape[3])

        tic.tic()
        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            col, row, height = pos2grid_id_3d(gs, cs, camera_height, p[0], p[1], p[2])
            if (
                col >= gs
                or row >= gs
                or height >= vh
                or col < 0
                or row < 0
                or height < 0
            ):
                continue

            rgb_px, rgb_py, rgb_pz = project_point(rgb_cam_mat, p_local)
            rgb_v = rgb[rgb_py, rgb_px, :]

            if max_id >= grid_feat.shape[0]:
                grid_feat = np.concatenate(
                    [
                        grid_feat,
                        np.zeros(
                            (grid_feat.shape[0], grid_feat.shape[1]), dtype=np.float32
                        ),
                    ],
                    axis=0,
                )
                grid_pos = np.concatenate(
                    [
                        grid_pos,
                        np.zeros(
                            (grid_pos.shape[0], grid_pos.shape[1]), dtype=np.int32
                        ),
                    ],
                    axis=0,
                )
                weight = np.concatenate(
                    [weight, np.zeros((weight.shape[0]), dtype=np.int32)], axis=0
                )
                grid_rgb = np.concatenate(
                    [   
                        grid_rgb,
                        np.zeros(
                            (grid_rgb.shape[0], grid_rgb.shape[1]), dtype=np.float32
                        ),
                    ],
                    axis=0,
                )

            radial_dist_sq = np.sum(np.square(p_local))
            sigma_sq = 0.6
            alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))

            px, py, pz = project_point(feat_cam_mat, p_local)
            if not (px < 0 or py < 0 or px >= pix_feats.shape[3] or py >= pix_feats.shape[2]):
                feat = pix_feats[0, :, py, px]
                occupied_id = occupied_ids[row, col, height]
                if occupied_id == -1:
                    occupied_ids[row, col, height] = max_id
                    grid_feat[max_id] = feat.flatten() * alpha
                    grid_rgb[max_id] = rgb_v
                    weight[max_id] += alpha
                    grid_pos[max_id] = [row, col, height]
                    max_id += 1
                else:
                    grid_feat[occupied_id] = (
                        grid_feat[occupied_id] * weight[occupied_id]
                        + feat.flatten() * alpha
                    ) / (weight[occupied_id] + alpha)
                    grid_rgb[occupied_id] = (
                        grid_rgb[occupied_id] * weight[occupied_id]
                        + rgb_v * alpha
                    ) / (weight[occupied_id] + alpha)
                    weight[occupied_id] += alpha

        tic.print_time("projecting features")

        mapped_iter_set.add(iter_i)
        if iter_i % 100 == 99:
            print(f"Temporarily saving {max_id} features at iter {iter_i}...")
            grid_feat = grid_feat[:max_id]
            grid_pos = grid_pos[:max_id]
            weight = weight[:max_id]
            save_concept_fusion_map(
                concept_fusion_map_save_path,
                grid_feat,
                grid_pos,
                weight,
                occupied_ids,
                list(mapped_iter_set),
            )

        et = time.time()
        pbar.set_description(f"Iteration time: {et-st}")
        pbar.update(1)

    print(f"Saving {max_id} features...")
    grid_feat = grid_feat[:max_id]
    grid_pos = grid_pos[:max_id]
    weight = weight[:max_id]
    grid_rgb = grid_rgb[:max_id]
    save_concept_fusion_map(
        concept_fusion_map_save_path,
        grid_feat,
        grid_pos,
        weight,
        occupied_ids,
        list(mapped_iter_set),
        grid_rgb
    )


@hydra.main(
    version_base=None,
    config_path="../../config/habitat",
    config_name="vlmaps_lseg_config.yaml",
)
def main(config: DictConfig):
    data_root_dir = config.data_paths.vlmaps_data_dir
    data_dir_base = config.data_dir_base
    data_dir = os.path.join(data_root_dir, data_dir_base)
    camera_height = config.camera_height
    cs = config.cs
    gs = config.gs
    depth_sample_rate = config.depth_sample_rate
    create_vlmaps_3d_batch(data_dir, camera_height, cs, gs, depth_sample_rate)


if __name__ == "__main__":
    main()