import os
import numpy as np
import h5py
from scipy.spatial.distance import cdist
from scipy.special import softmax
import cv2
import clip
from utils.clip_utils import *
from typing import List, Dict, Tuple, Set


def fuse_local_global_features(
    global_feats: np.ndarray,
    local_feats: np.ndarray,
    h: int,
    w: int,
    local_bboxes: List[np.ndarray],
    masks: np.ndarray,
):
    """
    local_bboxes: [(rmin, cmin, rmax, cmax), ]
    """
    l2g_sim = get_features_similarity(local_feats, global_feats.reshape((1, -1)))
    l2l_sim = get_features_similarity(local_feats, local_feats)
    sim_diag = np.eye(l2l_sim.shape[0]) * np.diag(l2l_sim)
    l_sim = np.mean(l2l_sim - sim_diag, axis=1)
    sim = l2g_sim.flatten() + l_sim.flatten()
    g_weight = softmax(sim)
    feat_dim = global_feats.shape[-1]
    del l2g_sim, l2l_sim, sim_diag, l_sim, sim
    feats = np.zeros((h, w, feat_dim))
    fuse_feats = global_feats.reshape((1, -1)) * g_weight.reshape(
        (-1, 1)
    ) + local_feats * (1 - g_weight).reshape((-1, 1))
    for ri, (feat, box) in enumerate(zip(fuse_feats, local_bboxes)):
        feat = feat / np.linalg.norm(feat)
        mask = masks == ri
        feats[mask] += feat
        feats[mask] /= np.linalg.norm(feats[mask], axis=-1, keepdims=True)
    return feats

def fuse_local_global_features_v2(
    global_feats: np.ndarray,
    local_feats: np.ndarray,
    h: int,
    w: int,
    local_bboxes: List[np.ndarray],
    masks: np.ndarray,
):
    """
    local_bboxes: [(rmin, cmin, rmax, cmax), ]
    """
    l2g_sim = get_features_similarity(local_feats, global_feats.reshape((1, -1)))
    l2l_sim = get_features_similarity(local_feats, local_feats)
    sim_diag = np.eye(l2l_sim.shape[0]) * np.diag(l2l_sim)
    l_sim = np.mean(l2l_sim - sim_diag, axis=1)
    sim = l2g_sim.flatten() + l_sim.flatten()
    g_weight = softmax(sim)
    feat_dim = global_feats.shape[-1]
    del l2g_sim, l2l_sim, sim_diag, l_sim, sim
    feats = np.zeros((h, w, feat_dim))
    fuse_feats = global_feats.reshape((1, -1)) * g_weight.reshape(
        (-1, 1)
    ) + local_feats * (1 - g_weight).reshape((-1, 1))
    for ri, (feat, box) in enumerate(zip(fuse_feats, local_bboxes)):
        feat = feat / np.linalg.norm(feat)
        mask = masks[ri] > 0
        feats[mask] += feat
        feats[mask] /= np.linalg.norm(feats[mask], axis=-1, keepdims=True)
    return feats


def load_bboxes(bboxes_path: str):
    bboxes = []
    with open(bboxes_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            bboxes.append([int(float(x)) for x in line.split(",")])
    return bboxes


def load_masks(mask_path: str):
    with open(mask_path, "rb") as f:
        mask = np.load(f)
    return mask


def compute_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def get_features_similarity(features_a: np.ndarray, features_b: np.ndarray):
    """
    features: (N, D) np.ndarray
    Return (N, N) distance matrix
    """
    print(features_a.shape, features_b.shape)
    similarity = features_a @ features_b.T
    return similarity


def generate_fused_features(
    img_path: str, bboxes_path: str, mask_path: str, clip_model, preprocess
):
    """
    Return (H, W, D) np.ndarray
    """
    img = cv2.imread(img_path)
    # img_rgb = img[:, :, ::-1]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    global_img_feat = get_img_feats(img, preprocess, clip_model).reshape((1, -1))
    bboxes = load_bboxes(bboxes_path)
    masks = load_masks(mask_path)
    all_local_feats = []
    if len(bboxes) == 0:
        return np.tile(global_img_feat.reshape((1,1,-1)), (img.shape[0], img.shape[1], 1))
    for box_i, box in enumerate(bboxes):
        if box[2] - box[0] < 50 or box[3] - box[1] < 50:
            continue
        local_img = img[box[0] : box[2], box[1] : box[3], :]
        local_img_feat = get_img_feats(local_img, preprocess, clip_model).reshape(
            (1, -1)
        )
        all_local_feats.append(local_img_feat)
    all_local_feats = np.concatenate(all_local_feats, axis=0)
    fused_feats = fuse_local_global_features(
        global_img_feat, all_local_feats, img.shape[0], img.shape[1], bboxes, masks
    )
    return fused_feats


def generate_fused_features_v2(
    img_path: str, mask_path: str, clip_model, clip_feat_dim, preprocess, batch_size: int=8
):
    """
    Return (H, W, D) np.ndarray
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    global_img_feat = get_img_feats(img, preprocess, clip_model).reshape((1, -1))
    with open(mask_path, "rb") as f:
        mask_pred = np.load(f)
    bboxes = []
    for i in range(mask_pred.shape[0]):
        mask_i = mask_pred[i]
        rows, cols = np.where(mask_i > 0)
        if len(rows) == 0:
            bboxes.append([0, 0, 0, 0])
            continue
        rmin, rmax = np.min(rows), np.max(rows)
        cmin, cmax = np.min(cols), np.max(cols)
        bboxes.append([rmin, cmin, rmax, cmax])

    assert len(bboxes) == mask_pred.shape[0]
    masks = load_masks(mask_path)
    if len(bboxes) == 0:
        return global_img_feat
    local_imgs = []
    for box_i, box in enumerate(bboxes):
        local_img = img[box[0] : box[2], box[1] : box[3], :]
        local_imgs.append(local_img)
    all_local_feats = get_imgs_feats_batch(
        local_imgs, preprocess, clip_model, clip_feat_dim, batch_size=batch_size
    )
    fused_feats = fuse_local_global_features_v2(
        global_img_feat, all_local_feats, img.shape[0], img.shape[1], bboxes, masks
    )
    return fused_feats


def save_efficient_map(save_path: str, map: np.ndarray):
    with h5py.File(save_path, "w") as f:
        f.create_dataset("map", data=map)
        pass


def load_efficient_map(map_path: str):
    with h5py.File(map_path, "r") as f:
        map = f["map"][:]
        return map


def save_concept_fusion_map(
    save_path: str,
    grid_feat: np.ndarray,
    grid_pos: np.ndarray,
    weight: np.ndarray,
    occupied_ids: np.ndarray,
    mapped_iter_list: Set[int],
    grid_rgb: np.ndarray = None,
):
    with h5py.File(save_path, "w") as f:
        f.create_dataset(
            "mapped_iter_list", data=np.array(mapped_iter_list, dtype=np.int32)
        )
        f.create_dataset("grid_feat", data=grid_feat)
        f.create_dataset("grid_pos", data=grid_pos)
        f.create_dataset("weight", data=weight)
        f.create_dataset("occupied_ids", data=occupied_ids)
        if grid_rgb is not None:
            f.create_dataset("grid_rgb", data=grid_rgb)


def load_concept_fusion_map(map_path: str):
    with h5py.File(map_path, "r") as f:
        mapped_iter_list = f["mapped_iter_list"][:].tolist()
        grid_feat = f["grid_feat"][:]
        grid_pos = f["grid_pos"][:]
        weight = f["weight"][:]
        occupied_ids = f["occupied_ids"][:]
        grid_rgb = None
        if "grid_rgb" in f:
            grid_rgb = f["grid_rgb"][:]
        return mapped_iter_list, grid_feat, grid_pos, weight, occupied_ids, grid_rgb
