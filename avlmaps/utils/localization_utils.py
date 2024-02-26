import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import cv2
import PIL
import torch
from tqdm import tqdm
import h5py
from matplotlib import cm

# from avlmaps.utils.mapping_utils import *

from hloc import (
    extractors,
    logger,
    extract_features,
    pairs_from_retrieval,
    match_features,
    matchers,
)
import pycolmap
from hloc.utils.base_model import dynamic_load

# from hloc.utils.tools import map_tensor
from third_party.SuperGluePretrainedNetwork.models.utils import (
    make_matching_plot_fast,
    frame2tensor,
)
from third_party.SuperGluePretrainedNetwork.models.matching import Matching
from avlmaps.utils.mapping_utils import (
    load_depth_npy,
    load_pose,
    load_real_pose,
    depth2pc,
    get_sim_cam_mat,
    cvt_pose_vec2tf,
)
from typing import Tuple, List, Dict


def resize_image(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


class ImageDataset(torch.utils.data.Dataset):
    default_conf = {
        "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
        "grayscale": False,
        "resize_max": None,
        "resize_force": False,
        "interpolation": "cv2_area",  # pil_linear is more accurate but slower
        # if the dataset is for reference images True, if it is query image False
        "reference": True,
    }

    def __init__(self, images_list, conf=default_conf, mode="color"):
        # self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.conf = conf
        self.mode = mode

        self.images_list = images_list
        self.is_path = False
        if isinstance(self.images_list[0], str):
            self.is_path = True

    def set_reference(self):
        self.conf["reference"] = True

    def set_query(self):
        self.conf["reference"] = False

    def get_path(self, idx):
        if self.is_path:
            return self.images_list[idx]
        raise Exception("images_list is not a list of paths but arrays.")

    def __getitem__(self, idx):
        if self.is_path:
            image_path = self.images_list[idx]
            name = image_path
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            name = f"{idx}".zfill(6)
            image = self.images_list[idx]
        if self.mode == "grayscale":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]

        if self.conf["resize_max"] and (self.conf["resize_force"] or max(size) > self.conf["resize_max"]):
            scale = self.conf["resize_max"] / max(size)
            size_new = tuple(int(round(x * scale)) for x in size)
            image = resize_image(image, size_new, self.conf["interpolation"])

        if self.conf["grayscale"] and self.mode == "grayscale":
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.0

        data = {
            "name": name,
            "image": image,
            "original_size": np.array(size),
        }
        return data

    def __len__(self):
        return len(self.images_list)


class HLocLocalizer:
    def __init__(self, features_dir):
        self.features_dir = features_dir
        os.makedirs(self.features_dir, exist_ok=True)
        self.images_dir = ""
        self.dataset_name = ""

        self.conf = extract_features.confs["netvlad"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = dynamic_load(extractors, self.conf["model"]["name"])
        self.model = Model(self.conf["model"]).eval().to(self.device)

        self.features_conf = extract_features.confs["superpoint_max"]
        self.match_conf = match_features.confs["superglue"]
        self.super_conf = {
            "superpoint": self.features_conf,
            "superglue": self.match_conf,
        }
        self.matching = Matching(self.super_conf).eval().to(self.device)

    def init_video_with_images_list(self, images_list):
        self.images_list = images_list

    def init_video_with_images_folder(
        self, images_dir, frame_sample_interval=50, key=lambda x: int(x.split(".")[0].split("_")[-1]), show=False
    ):
        self.frame_sample_interval = frame_sample_interval
        self.images_dir = images_dir
        image_paths = Path(images_dir)
        self.image_paths_list = sorted(
            [str(x) for x in list(image_paths.glob("*.png"))],
            key=key,
        )

    def init_depth_with_depth_folder(
        self,
        depths_dir,
        frame_sample_interval=50,
        key=lambda x: int(x.split(".")[0].split("_")[-1]),
        show=False,
        extension=".npy",
    ):
        self.depth_extension = extension
        self.frame_sample_interval = frame_sample_interval
        self.depths_dir = depths_dir
        depth_paths = Path(depths_dir)
        self.depth_paths_list = sorted(
            [str(x) for x in list(depth_paths.glob("*" + extension))],
            key=key,
        )

    def init_pose_with_pose_folder(
        self, poses_dir, frame_sample_interval=50, key=lambda x: int(x.split(".")[0].split("_")[-1])
    ):
        self.frame_sample_interval = frame_sample_interval
        self.poses_dir = poses_dir
        pose_paths = Path(poses_dir)
        self.pose_paths_list = sorted(
            [str(x) for x in list(pose_paths.glob("*.txt"))],
            key=key,
        )

    def init_pose_with_pose_file(self, pose_path: Path, frame_sample_interval=50):
        self.pose_list = np.loadtxt(pose_path)
        self.pose_list = [cvt_pose_vec2tf(pose) for pose in self.pose_list]
        self.frame_sample_interval = frame_sample_interval

    def init_video_with_video_path(self, video_path):
        cap = cv2.VideoCapture(video_path)
        self.images_list = []
        while True:
            success, frame = cap.read()
            if success:
                self.images_list.append(frame)

    def subsample_images(self):
        raise NotImplementedError

    @torch.no_grad()
    def detect_images_keypoints(self, images_list):
        outputs = []
        imgs_tensor_shapes = []
        pbar = tqdm(enumerate(images_list))
        for img_i, img in pbar:
            pbar.set_description(f"Detecting keypoints for image {img_i}")
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_tensor = frame2tensor(img_gray, self.device)
            output = self.matching.superpoint({"image": img_tensor})
            outputs.append(output)
            imgs_tensor_shapes.append(img_tensor.shape)

        return outputs, imgs_tensor_shapes

    @torch.no_grad()
    def match_keypoints_with_detection_output(self, kpts_dict0, kpts_dict1, image0_tensor_shape, image1_tensor_shape):
        """
        Input:
        kpts_dict0, kpts_dict1: the output dictionary from keypoints detection algorithm SuperPoint.
        Each dict contains keys "keypoints", "scores", and "descriptors"
        image0_shape: useful for matching
        """
        keys = ["keypoints", "scores", "descriptors"]
        last_data = {k + "0": kpts_dict0[k] for k in keys}
        last_data["image0"] = torch.zeros(image0_tensor_shape)
        last_data["image1"] = torch.zeros(image1_tensor_shape)
        last_data = {**last_data, **{k + "1": kpts_dict1[k] for k in keys}}
        pred = self.matching(last_data)

        kpts0 = last_data["keypoints0"][0].cpu().numpy()
        kpts1 = last_data["keypoints1"][0].cpu().numpy()
        matches = pred["matches0"][0].cpu().numpy()
        confidence = pred["matching_scores0"][0].cpu().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        return kpts0, kpts1, mkpts0, mkpts1, confidence, valid

    @torch.no_grad()
    def match_keypoints(self, img0, img1, vis: bool = False):
        """match 0 to 1"""
        keys = ["keypoints", "scores", "descriptors"]
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

        img0_tensor = frame2tensor(img0, self.device)
        img1_tensor = frame2tensor(img1, self.device)
        last_data = self.matching.superpoint({"image": img0_tensor})
        last_data = {k + "0": last_data[k] for k in keys}
        last_data["image0"] = img0_tensor
        last_frame = img0
        last_image_id = 0

        pred = self.matching({**last_data, "image1": img1_tensor})
        kpts0 = last_data["keypoints0"][0].cpu().numpy()
        kpts1 = pred["keypoints1"][0].cpu().numpy()
        matches = pred["matches0"][0].cpu().numpy()
        confidence = pred["matching_scores0"][0].cpu().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        if vis:
            self.visulize_keypoints(img0, img1, kpts0, kpts1, mkpts0, mkpts1, confidence, valid)
        return mkpts0, mkpts1, confidence

    def visulize_keypoints(self, img0, img1, kpts0, kpts1, mkpts0, mkpts1, confidence, valid):
        if len(img0.shape) == 3 and len(img1.shape) == 3:
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

        color = cm.jet(confidence[valid])
        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
            "Matches: {}".format(len(mkpts0)),
        ]
        k_thresh = self.matching.superpoint.config["keypoint_threshold"]
        m_thresh = self.matching.superglue.config["match_threshold"]

        small_text = [
            "Keypoint Threshold: {:.4f}".format(k_thresh),
            "Match Threshold: {:.2f}".format(m_thresh),
        ]
        vis_img = make_matching_plot_fast(
            img0,
            img1,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            path=None,
            show_keypoints=True,
            small_text=small_text,
        )

        cv2.imshow("matching results", vis_img)
        cv2.waitKey(0)

    @torch.no_grad()
    def compute_global_descriptor(
        self,
        images_list,
        as_half=False,
        reference=True,
        overwrite=True,
        descriptor_filename="",
    ):
        """features_path: either for loading (if features already exists) or for saving"""
        if not self.dataset_name:
            self.dataset_name = descriptor_filename
        self.descriptor_file_prefix = self.dataset_name + "_"
        if self.dataset_name == "":
            self.descriptor_file_prefix = "tmp_"

        images_dataset = ImageDataset(images_list)
        loader = torch.utils.data.DataLoader(images_dataset, num_workers=1)
        loader_iter = iter(loader)
        if reference:
            features_path = os.path.join(
                self.features_dir,
                self.descriptor_file_prefix + "reference_features.h5df",
            )
            self.ref_features_path = features_path
            print(f"reading reference global features at {features_path}")
        else:
            features_path = os.path.join(self.features_dir, self.descriptor_file_prefix + "query_features.h5df")
            self.query_features_path = features_path
            print(f"reading query global features at {features_path}")

        p = Path(features_path)
        # check if partial descriptors are computed
        starting_id = -1
        if p.exists() and overwrite:
            with h5py.File(str(features_path), "w") as fd:
                pass

        pbar = tqdm(total=len(loader))
        for id in range(len(loader)):
            pbar.set_description(f"Creating VisualMap Frame {id:06}")
            # skip computed descriptors
            if images_dataset.is_path:
                name = images_dataset.get_path(id)

                # test if the features are already stored
                with h5py.File(str(features_path), "a") as fd:
                    if name in fd:
                        pbar.set_description(f"skipping features {name}")
                        pbar.update(1)
                        continue

            else:
                name = f"{id}".zfill(6)
            data = next(loader_iter)
            print(data["image"].shape)

            pred = self.model({"image": data["image"].to(self.device, non_blocking=True)})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

            pred["image_size"] = original_size = data["original_size"][0].numpy()
            if "keypoints" in pred:
                size = np.array(data["image"].shape[-2:][::-1])
                scales = (original_size / size).astype(np.float32)
                pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
                # add keypoint uncertainties scaled to the original resolution
                uncertainty = getattr(self.model, "detection_noise", 1) * scales.mean()

            if as_half:
                for k in pred:
                    dt = pred[k].dtype
                    if (dt == np.float32) and (dt != np.float16):
                        pred[k] = pred[k].astype(np.float16)

            with h5py.File(str(features_path), "a") as fd:
                try:
                    if name in fd:
                        del fd[name]
                    pbar.set_description(f"saving features {name}")
                    pbar.update(1)
                    grp = fd.create_group(name)
                    for k, v in pred.items():
                        grp.create_dataset(k, data=v)
                    if "keypoints" in pred:
                        grp["keypoints"].attrs["uncertainty"] = uncertainty
                except OSError as error:
                    if "No space left on device" in error.args[0]:
                        logger.error(
                            "Out of disk space: storing features on disk can take "
                            "significant space, did you enable the as_half flag?"
                        )
                        del grp, fd[name]
                    raise error

            del pred

        return True

    def localize_agent(self, rgb_obs):
        """Return the most similar frame's id to the query"""
        # setup features loading or saving paths
        os.makedirs(self.features_dir, exist_ok=True)
        reference_features_path = os.path.join(
            self.features_dir, self.descriptor_file_prefix + "reference_features.h5df"
        )
        query_features_path = os.path.join(self.features_dir, self.descriptor_file_prefix + "query_features.h5df")
        # print(f"reading reference global features at {reference_features_path}")
        # print(f"reading query global features at {query_features_path}")

        # compute reference features
        # self.compute_global_descriptor(
        #     self.image_paths_list, reference=True, overwrite=False
        # )
        # compute query features
        if isinstance(rgb_obs, list):
            self.compute_global_descriptor(rgb_obs, reference=False, overwrite=True)
        else:
            self.compute_global_descriptor([rgb_obs], reference=False, overwrite=True)

        # pick the most relevant frame in reference to the query
        # names = [str(x).zfill(6) for x in range(len(self.image_paths_list))]
        names = [x for x in self.image_paths_list]
        ref_desc = pairs_from_retrieval.get_descriptors(names, reference_features_path)
        query_desc = pairs_from_retrieval.get_descriptors(["000000"], query_features_path)
        # print("ref_desc:", ref_desc)
        # print("query_desc:", query_desc)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sim = torch.einsum("id,jd->ij", query_desc.to(device), ref_desc.to(device))
        # print("scores: ")
        # for i, s in enumerate(sim):
        #     print(i, s)
        top_id = torch.argmax(sim, dim=1)
        max_sim = sim[0, top_id].cpu().numpy()
        top_id = top_id.cpu().numpy()
        # print(f"localize to frame {top_id[0]} with score {max_sim[0]}")
        self.current_frame_id = top_id[0]
        return top_id[0], max_sim[0]

    def _get_relative_pose_with_depth(
        self,
        rgb_ref,
        rgb_obs,
        depth_ref,
        ref_intr_mat=None,
        query_intr_mat=None,
        vis: bool = False,
    ):
        mkpts0, mkpts1, confidence = self.match_keypoints(rgb_ref, rgb_obs, vis=vis)  # mkpts0 (N, 2)
        if mkpts0.shape[0] < 100:
            return None
        h, w = depth_ref.shape
        if ref_intr_mat is None:
            ref_intr_mat = get_sim_cam_mat(h, w)
        pc, mask = depth2pc(depth_ref, intr_mat=ref_intr_mat)

        mask = mask.reshape((h, w))
        pc = pc.reshape((3, h, w))

        mkpts0_int = mkpts0.astype(np.int32)
        mkpts0_3d = pc[:, mkpts0_int[:, 1], mkpts0_int[:, 0]]
        mkpts0_mask = mask[mkpts0_int[:, 1], mkpts0_int[:, 0]]
        mkpts0_3d = mkpts0_3d[:, mkpts0_mask]
        mkpts1 = mkpts1[mkpts0_mask, :]

        query_intr_mat = query_intr_mat
        if query_intr_mat is None:
            query_intr_mat = get_sim_cam_mat(h, w)
        camera = pycolmap.Camera(
            model="SIMPLE_PINHOLE",
            width=w,
            height=h,
            params=[query_intr_mat[0, 0], query_intr_mat[0, 2], query_intr_mat[1, 2]],
        )
        conf = {
            "estimation": {"ransac": {"max_error": 12}},
            "refinement": {"refine_focal_length": False, "refine_extra_params": False},
        }

        mkpts1_list = np.array([x.reshape((2, 1)) for x in mkpts1]).reshape((-1, 2))
        mkpts0_3d_list = np.array([x.reshape((3, 1)) for x in mkpts0_3d.T]).reshape((-1, 3))

        ret = pycolmap.absolute_pose_estimation(
            mkpts1_list,
            mkpts0_3d_list,
            camera,
            estimation_options=conf.get("estimation", {}),
            refinement_options=conf.get("refinement", {}),
        )

        # t = ret["tvec"]
        # quat = ret["qvec"]  # wxyz
        t = ret["cam_from_world"].translation
        rot = np.array(ret["cam_from_world"].rotation.matrix())
        # print("t, rot", t, rot)
        # print(quat)
        transform = np.eye(4)
        # r = R.from_quat(np.array([quat[1], quat[2], quat[3], quat[0]]))  # takes xyzw as input
        # r = R.from_quat(quat)  # takes xyzw as input
        # rot = r.as_matrix()

        transform = np.eye(4)
        transform[:3, 3] = t
        transform[:3, :3] = rot
        transform = np.linalg.inv(transform)
        return transform

    def localize_agent_with_depth(
        self,
        rgb_obs: np.ndarray,
        ref_intr_mat: np.ndarray = None,
        query_intr_mat: np.ndarray = None,
        depth_scale: float = 1.0,
        vis: bool = False,
    ) -> Tuple[int, np.ndarray]:
        """
        depth_scale: real_depth_value / saved_depth_value, usually 0.001, default 1.0
        """
        img_id, max_sim = self.localize_agent(rgb_obs)
        print("img_id: ", img_id)
        ref_img_path = self.image_paths_list[img_id]
        ref_img = cv2.imread(ref_img_path)
        if vis:
            cv2.imshow("ref_img", ref_img)
            cv2.waitKey()
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        ref_depth_path = self.depth_paths_list[img_id]
        # print("image path: ", ref_img_path)
        # print("depth path: ", ref_depth_path)
        if self.depth_extension == ".npy":
            ref_depth = load_depth_npy(ref_depth_path).astype(float)
        elif self.depth_extension == ".png":
            ref_depth = cv2.imread(ref_depth_path, cv2.IMREAD_ANYDEPTH).astype(float)
        # cv2.imshow("depth", ref_depth * depth_scale * 0.1)
        ref_depth *= depth_scale
        # print("depth min, max: ", np.min(ref_depth), np.max(ref_depth))
        # print("[0, 0] depth: ", ref_depth[0, 0])
        transform = self._get_relative_pose_with_depth(
            ref_img,
            rgb_obs,
            ref_depth,
            ref_intr_mat=ref_intr_mat,
            query_intr_mat=query_intr_mat,
            vis=vis,
        )
        # print("transform:\n", transform)
        if transform is None:
            return -1, None
        return img_id, transform


def get_frames_tfs(
    localizer: HLocLocalizer,
    segment_frames: List[np.ndarray],
    pose_list: List[np.ndarray],
    init_tf_inv: np.ndarray,
    ref_cam_mat: np.ndarray = None,
    query_cam_mat: np.ndarray = None,
    masked_obst: np.ndarray = None,
    vis: bool = False,
) -> List[List[np.ndarray]]:
    tfs = []
    for frame_i, frame in enumerate(segment_frames):
        ref_img_id, transform = localizer.localize_agent_with_depth(
            frame, ref_intr_mat=ref_cam_mat, query_intr_mat=query_cam_mat, vis=vis
        )
        if ref_img_id == -1:
            tfs.append(None)
            continue

        tf = init_tf_inv @ get_cam_pose_habitat(pose_list[ref_img_id])
        query_tf = tf @ transform
        tfs.append(query_tf)

        if masked_obst is not None:
            refcol, refrow = pos2grid_id(1000, 0.05, tf[0, 3], tf[2, 3])
            col, row = pos2grid_id(1000, 0.05, query_tf[0, 3], query_tf[2, 3])
            masked_obst = cv2.circle(masked_obst, (int(refcol), int(refrow)), 2, (0, 0, 255), -1)  # red (BGR)
            masked_obst = cv2.circle(masked_obst, (int(col), int(row)), 2, (255, 0, 0), -1)  # blue (BGR)
            cv2.imshow("obstacles", masked_obst)
            cv2.waitKey(1)

    return tfs


def get_frames_tfs_real(
    localizer: HLocLocalizer,
    segment_frames: List[np.ndarray],
    pose_list: List[np.ndarray],
    init_tf_inv: np.ndarray,
    ref_cam_mat: np.ndarray = None,
    query_cam_mat: np.ndarray = None,
    masked_obst: np.ndarray = None,
    vis: bool = False,
) -> List[List[np.ndarray]]:
    tfs = []
    for frame_i, frame in enumerate(segment_frames):
        ref_img_id, transform = localizer.localize_agent_with_depth(
            frame, ref_intr_mat=ref_cam_mat, query_intr_mat=query_cam_mat, vis=vis, depth_scale=0.001
        )
        if ref_img_id == -1:
            tfs.append(None)
            continue
        print("transform:", transform)

        ref_cam_pose = load_real_pose(localizer.pose_paths_list[ref_img_id])
        tf = init_tf_inv @ ref_cam_pose
        query_tf = tf @ transform
        tfs.append(query_tf)

    return tfs


def get_cam_pose_habitat(pose_path, camera_height=1.5):

    pos, rot = load_pose(pose_path)

    rot_ro_cam = np.eye(3)
    rot_ro_cam[1, 1] = -1
    rot_ro_cam[2, 2] = -1
    rot = rot @ rot_ro_cam
    pos[1] += camera_height

    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = pos.reshape(-1)
    return pose


def save_hab_tf(save_path: str, tf: np.ndarray):
    with open(save_path, "w") as f:
        if tf is None:
            return
        line = ",".join([str(x) for x in tf.flatten().tolist()])
        f.write(line)


def save_video_frame_tfs(tfs: List[np.ndarray], save_dir: str):
    for tf_i, tf in enumerate(tfs):
        save_path = os.path.join(save_dir, f"{tf_i:06}.txt")
        save_hab_tf(save_path, tf)
