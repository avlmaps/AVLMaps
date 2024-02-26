import os
from pathlib import Path
import pickle

import numpy as np
import torchvision as tv
import librosa as lb
from tqdm import tqdm


from avlmaps.utils.clip_utils import *
from avlmaps.utils.mapping_utils import load_pose, cvt_pose_vec2tf
from avlmaps.utils.audio_utils import (
    extract_audio_from_video,
    segment_audio_with_silence,
    convert_time_ranges_to_frame_ranges,
    create_audio_dictionary,
    encode_audio_batch,
    setup_audio_paths,
)
from avlmaps.audioclip.model.audioclip import AudioCLIP
from avlmaps.audioclip.utils.transforms import ToTensor1D, RandomCrop, RandomPadding
from typing import List, Dict, Tuple


def create_audio_map_batch(
    data_dir: str,
    aclp: AudioCLIP = None,
    audio_transforms: tv.transforms.Compose = None,
    sample_rate: int = 44100,
    silence_duration_s: float = 1,
    silence_thres: int = 0,
    fps: float = 20,
    difficulty_level: str = "level_1",
    manual_mode: bool = False,
    seq_num: int = None,
):
    scene_name = data_dir.split("/")[-1].split("_")[0]

    # check all audio videos
    audio_video_dir, seq_dirs = setup_audio_paths(data_dir)
    seq_dirs = seq_dirs[:seq_num] if seq_num is not None else seq_dirs

    seqs_segments_frames_tfs = []
    seqs_audio_features = []

    if aclp is None:
        MODEL_FILENAME = "AudioCLIP-Partial-Training.pt"
        model_path = Path(__file__).parent.parent / "audioclip" / "assets" / MODEL_FILENAME
        aclp = AudioCLIP(pretrained=model_path.as_posix()).eval()
        audio_transforms = tv.transforms.Compose(
            [
                ToTensor1D(),
                RandomPadding(out_len=220500, train=False),
                RandomCrop(out_len=220500, train=False),
            ]
        )
    manual_str = ""
    if manual_mode:
        manual_str = "_manual"
    scene_data = []
    pbar = tqdm(enumerate(seq_dirs), total=len(seq_dirs), leave=False)
    for seq_i, seq_dir in pbar:
        pbar.set_description(desc=f"Creating SoundMap Sequence {seq_i:03}")

        meta_path = os.path.join(seq_dir, f"range_and_audio{manual_str}_meta_{difficulty_level}.txt")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r") as f:
            for line in f:
                row = line.strip("\n").split(",")
                scene_data.append(row[-2:])
        audio_video_path = os.path.join(seq_dir, f"output_with_audio{manual_str}_{difficulty_level}.mp4")
        extract_audio_from_video(audio_video_path, f"/tmp/{scene_name}_{seq_i}.wav")

        # detect audio segment time ranges
        segment_time_ranges, segment_tracks = segment_audio_with_silence(
            f"/tmp/{scene_name}_{seq_i}.wav",
            silence_duration_s=silence_duration_s,
            silence_thres=silence_thres,
            sample_rate=sample_rate,
        )  # generate AudioCLIP features for each audio segment
        if len(segment_tracks) == 0:
            continue
        segment_tracks = [x.astype(np.float32) * 32768.0 for x in segment_tracks]
        audio_features = encode_audio_batch(segment_tracks, aclp, audio_transforms, sample_rate)

        audio_features = [audio_features[i] for i in range(audio_features.shape[0])]
        seqs_audio_features.extend(audio_features)

        # convert all time ranges into frame ranges
        segment_frame_ranges = convert_time_ranges_to_frame_ranges(segment_time_ranges, fps)

        st_frames = [x[0] for x in segment_frame_ranges]
        ed_frames = [x[1] for x in segment_frame_ranges]

        audio_pos_list = []
        pose_path = Path(seq_dir) / "poses.txt"
        seq_pose_list = np.loadtxt(pose_path)
        # seq_pose_dir = os.path.join(seq_dir, "pose")
        # seq_pose_list = [os.path.join(seq_pose_dir, x) for x in sorted(os.listdir(seq_pose_dir))]
        for seg_i, seg_range in enumerate(segment_frame_ranges):
            seg_pose_list = seq_pose_list[seg_range[0] : seg_range[1]]
            seg_pos_list = []
            for seg_pose_i, posevec in enumerate(seg_pose_list):
                pose = cvt_pose_vec2tf(posevec)
                seg_pos_list.append(pose[:3, 3].flatten())
            audio_pos_list.append(seg_pos_list)

        seqs_segments_frames_tfs.extend(audio_pos_list)

    # for content in scene_data:
    #     print(content)
    # create the dictionary for the audio features and the locations
    audio_database = create_audio_dictionary(seqs_audio_features, seqs_segments_frames_tfs)

    # save the dictionary
    audio_data_save_path = os.path.join(data_dir, "audio_video", f"audio_data{manual_str}_{difficulty_level}.pkl")
    if not audio_data_save_path.endswith(".pkl"):
        audio_data_save_path += ".pkl"
    with open(audio_data_save_path, "wb") as f:
        pickle.dump(audio_database, f)


def create_audio_map_statistics(
    data_dir: str,
    difficulty_level: str = "level_1",
    manual_mode: bool = False,
    seq_num: int = None,
):
    audio_video_dir, seq_dirs = setup_audio_paths(data_dir)
    seq_dirs = seq_dirs[:seq_num] if seq_num is not None else seq_dirs

    seqs_segments_frames_tfs = []
    seqs_audio_features = []

    manual_str = ""
    if manual_mode:
        manual_str = "_manual"
    scene_data = []
    for seq_i, seq_dir in enumerate(seq_dirs):
        meta_path = os.path.join(seq_dir, f"range_and_audio{manual_str}_meta_{difficulty_level}.txt")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r") as f:
            for line in f:
                row = line.strip("\n").split(",")
                scene_data.append(row[-2:])

    save_path = os.path.join(audio_video_dir, f"audio_map_statistics{manual_str}_{difficulty_level}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(scene_data, f)


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


def create_audio_map_batch_real(
    data_dir: str,
    time_ranges: List[List[List[float]]],
    sample_rate: int = 44100,
    # silence_duration_s: float = 1,
    fps: float = 30,
    # difficulty_level: str = "level_1",
    # manual_mode: bool = False,
    # seq_num: int = None,
):
    scene_name = data_dir.split("/")[-1].split("_")[0]
    rgb_list, depth_list, pose_list, map_path, obs_path = setup_paths(data_dir)

    pos_ro_cam, rot_ro_cam = load_pose(os.path.join(data_dir, "rob2cam_tf.txt"))
    rob2cam_tf = np.eye(4)
    rob2cam_tf[:3, :3] = rot_ro_cam
    rob2cam_tf[:3, 3] = pos_ro_cam.reshape(-1)
    cam2rob_tf = np.linalg.inv(rob2cam_tf)

    tf_list = []
    # init_tf = get_cam_pose_habitat(pose_list[0])  # z backward, y upward, x to the right
    cam_init_tf = load_real_pose(pose_list[0])
    cam_init_tf_inv = np.linalg.inv(cam_init_tf)
    base_init_tf = cam_init_tf @ cam2rob_tf
    base_init_tf_inv = np.linalg.inv(base_init_tf)

    # check all audio videos
    audio_video_dir, seq_dirs = setup_audio_paths(data_dir)

    seqs_segments_frames_tfs = []
    seqs_audio_features = []

    MODEL_FILENAME = "AudioCLIP-Partial-Training.pt"
    model_path = os.path.join("audioclip/assets/", MODEL_FILENAME)
    aclp = AudioCLIP(pretrained=model_path).eval()
    audio_transforms = tv.transforms.Compose(
        [
            ToTensor1D(),
            RandomPadding(out_len=220500, train=False),
            RandomCrop(out_len=220500, train=False),
        ]
    )
    scene_data = []
    for seq_i, seq_dir in enumerate(seq_dirs):
        print("########################")

        # meta_path = os.path.join(
        #     seq_dir, f"range_and_audio{manual_str}_meta_{difficulty_level}.txt"
        # )
        # if not os.path.exists(meta_path):
        #     continue
        # with open(meta_path, "r") as f:
        #     for line in f:
        #         row = line.strip("\n").split(",")
        #         scene_data.append(row[-2:])
        #         print(row[-2:])
        audio_video_path = os.path.join(seq_dir, f"audio_video.mp4")
        extract_audio_from_video(audio_video_path, f"/tmp/{scene_name}_{seq_i}.wav")

        # detect audio segment time ranges
        # segment_time_ranges, segment_tracks = segment_audio_with_silence(
        #     f"/tmp/{scene_name}_{seq_i}.wav",
        #     silence_duration_s=silence_duration_s,
        #     silence_thres=silence_thres,
        #     sample_rate=sample_rate,
        # )  # generate AudioCLIP features for each audio segment
        segment_time_ranges = time_ranges[seq_i]
        segment_frame_ranges = convert_time_ranges_to_frame_ranges(segment_time_ranges, fps)
        audio, sr = lb.load(f"/tmp/{scene_name}_{seq_i}.wav", sr=sample_rate)
        audio = nr.reduce_noise(y=audio, sr=sr)
        segment_tracks = []
        for seg_i, (start, end) in enumerate(segment_frame_ranges):
            segment = audio[start:end]
            segment_tracks.append(segment)

        if len(segment_tracks) == 0:
            continue
        segment_tracks = [x.astype(np.float32) * 32768.0 for x in segment_tracks]
        print(len(segment_tracks))

        audio_features = encode_audio_batch(segment_tracks, aclp, audio_transforms, sample_rate)

        audio_features = [audio_features[i] for i in range(audio_features.shape[0])]
        seqs_audio_features.extend(audio_features)

        # convert all time ranges into frame ranges
        segment_frame_ranges = convert_time_ranges_to_frame_ranges(segment_time_ranges, fps)

        st_frames = [x[0] for x in segment_frame_ranges]
        ed_frames = [x[1] for x in segment_frame_ranges]

        audio_pos_list = []
        seq_pose_dir = os.path.join(seq_dir, "pose")
        seq_pose_list = [os.path.join(seq_pose_dir, x) for x in sorted(os.listdir(seq_pose_dir))]
        for seg_i, seg_range in enumerate(segment_frame_ranges):
            seg_pose_list = seq_pose_list[seg_range[0] : seg_range[1]]
            seg_pos_list = []
            for seg_pose_i, pose_path in enumerate(seg_pose_list):
                cam_tf = load_real_pose(pose_path)
                base_tf = cam_tf @ cam2rob_tf
                seg_pos_list.append(base_tf[:3, 3].flatten())
            audio_pos_list.append(seg_pos_list)

        seqs_segments_frames_tfs.extend(audio_pos_list)

    # for content in scene_data:
    #     print(content)
    # create the dictionary for the audio features and the locations
    audio_database = create_audio_dictionary(seqs_audio_features, seqs_segments_frames_tfs)

    # save the dictionary
    audio_data_save_path = os.path.join(data_dir, "audio_video", f"audio_data.pkl")
    if not audio_data_save_path.endswith(".pkl"):
        audio_data_save_path += ".pkl"
    with open(audio_data_save_path, "wb") as f:
        pickle.dump(audio_database, f)


def create_audio_map_batch_real_with_clip(
    data_dir: str,
    time_ranges: List[List[List[float]]],
    audio_paths_list: List[List[str]],
    sample_rate: int = 44100,
    # silence_duration_s: float = 1,
    fps: float = 30,
    # difficulty_level: str = "level_1",
    # manual_mode: bool = False,
    # seq_num: int = None,
):
    scene_name = data_dir.split("/")[-1].split("_")[0]
    rgb_list, depth_list, pose_list, map_path, obs_path = setup_paths(data_dir)

    pos_ro_cam, rot_ro_cam = load_pose(os.path.join(data_dir, "rob2cam_tf.txt"))
    rob2cam_tf = np.eye(4)
    rob2cam_tf[:3, :3] = rot_ro_cam
    rob2cam_tf[:3, 3] = pos_ro_cam.reshape(-1)
    cam2rob_tf = np.linalg.inv(rob2cam_tf)

    tf_list = []
    # init_tf = get_cam_pose_habitat(pose_list[0])  # z backward, y upward, x to the right
    cam_init_tf = load_real_pose(pose_list[0])
    cam_init_tf_inv = np.linalg.inv(cam_init_tf)
    base_init_tf = cam_init_tf @ cam2rob_tf
    base_init_tf_inv = np.linalg.inv(base_init_tf)

    # check all audio videos
    audio_video_dir, seq_dirs = setup_audio_paths(data_dir)

    seqs_segments_frames_tfs = []
    seqs_audio_features = []

    MODEL_FILENAME = "AudioCLIP-Partial-Training.pt"
    model_path = os.path.join("audioclip/assets/", MODEL_FILENAME)
    aclp = AudioCLIP(pretrained=model_path).eval()
    audio_transforms = tv.transforms.Compose(
        [
            ToTensor1D(),
            RandomPadding(out_len=220500, train=False),
            RandomCrop(out_len=220500, train=False),
        ]
    )
    scene_data = []
    for seq_i, seq_dir in enumerate(seq_dirs):
        print("########################")
        print("seq_dir: ", seq_dir)

        seq_pos_ro_cam, seq_rot_ro_cam = load_pose(os.path.join(seq_dir, "rob2cam_tf.txt"))
        seq_rob2cam_tf = np.eye(4)
        seq_rob2cam_tf[:3, :3] = seq_rot_ro_cam
        seq_rob2cam_tf[:3, 3] = seq_pos_ro_cam.reshape(-1)
        seq_cam2rob_tf = np.linalg.inv(seq_rob2cam_tf)

        # meta_path = os.path.join(
        #     seq_dir, f"range_and_audio{manual_str}_meta_{difficulty_level}.txt"
        # )
        # if not os.path.exists(meta_path):
        #     continue
        # with open(meta_path, "r") as f:
        #     for line in f:
        #         row = line.strip("\n").split(",")
        #         scene_data.append(row[-2:])
        #         print(row[-2:])
        # audio_video_path = os.path.join(seq_dir, f"audio_video.mp4")
        # extract_audio_from_video(audio_video_path, f"/tmp/{scene_name}_{seq_i}.wav")

        # detect audio segment time ranges
        # segment_time_ranges, segment_tracks = segment_audio_with_silence(
        #     f"/tmp/{scene_name}_{seq_i}.wav",
        #     silence_duration_s=silence_duration_s,
        #     silence_thres=silence_thres,
        #     sample_rate=sample_rate,
        # )  # generate AudioCLIP features for each audio segment
        segment_time_ranges = time_ranges[seq_i]
        seq_audio_paths = audio_paths_list[seq_i]
        segment_tracks = []
        for audio_path in seq_audio_paths:
            audio, sr = lb.load(audio_path, sr=sample_rate)
            segment_tracks.append(audio)

        segment_frame_ranges = convert_time_ranges_to_frame_ranges(segment_time_ranges, fps)
        # audio, sr = lb.load(f"/tmp/{scene_name}_{seq_i}.wav", sr=sample_rate)
        # segment_tracks = []
        # for seg_i, (start, end) in enumerate(segment_frame_ranges):
        #     segment = audio[start: end]
        #     segment_tracks.append(segment)

        if len(segment_tracks) == 0:
            continue
        segment_tracks = [x.astype(np.float32) * 32768.0 for x in segment_tracks]
        print(len(segment_tracks))
        audio_features = encode_audio_batch(segment_tracks, aclp, audio_transforms, sample_rate)

        audio_features = [audio_features[i] for i in range(audio_features.shape[0])]
        seqs_audio_features.extend(audio_features)

        # convert all time ranges into frame ranges
        segment_frame_ranges = convert_time_ranges_to_frame_ranges(segment_time_ranges, fps)

        st_frames = [x[0] for x in segment_frame_ranges]
        ed_frames = [x[1] for x in segment_frame_ranges]

        audio_pos_list = []
        seq_pose_dir = os.path.join(seq_dir, "pose")
        seq_pose_list = [os.path.join(seq_pose_dir, x) for x in sorted(os.listdir(seq_pose_dir))]
        for seg_i, seg_range in enumerate(segment_frame_ranges):
            seg_pose_list = seq_pose_list[seg_range[0] : seg_range[1]]
            seg_pos_list = []
            for seg_pose_i, pose_path in enumerate(seg_pose_list):
                cam_tf = load_real_pose(pose_path)
                base_tf = cam_tf @ seq_cam2rob_tf
                seg_pos_list.append(base_tf[:3, 3].flatten())
            audio_pos_list.append(seg_pos_list)

        seqs_segments_frames_tfs.extend(audio_pos_list)

    # for content in scene_data:
    #     print(content)
    # create the dictionary for the audio features and the locations
    audio_database = create_audio_dictionary(seqs_audio_features, seqs_segments_frames_tfs)

    # save the dictionary
    audio_data_save_path = os.path.join(data_dir, "audio_video", f"audio_data.pkl")
    if not audio_data_save_path.endswith(".pkl"):
        audio_data_save_path += ".pkl"
    with open(audio_data_save_path, "wb") as f:
        pickle.dump(audio_database, f)
