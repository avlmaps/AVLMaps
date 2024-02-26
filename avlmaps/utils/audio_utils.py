import os
from collections import defaultdict
import pathlib
import wav2clip
import tqdm
import cv2
import numpy as np
import torchvision as tv
import clip
from scipy.spatial.distance import cdist
from omegaconf import DictConfig
import torch
import librosa as lb
import soundfile
from typing import List, Tuple, Dict, Set

from avlmaps.utils.clip_utils import get_text_feats
from avlmaps.utils.esc50_utils import ESC50Meta
from avlmaps.utils.category_assigner import CategoryAssigner
from avlmaps.dataloader.habitat_dataloader import VLMapsDataloaderHabitat
from avlmaps.utils.mapping_utils import load_pose
from avlmaps.audioclip.model.audioclip import AudioCLIP
from avlmaps.audioclip.utils.transforms import ToTensor1D, RandomCrop, RandomPadding


def generate_video_from_rgbs(video_save_path: str, rgb_dir: str, fps: float):
    prefix = sorted(os.listdir(rgb_dir))[0].split("_")[0]
    # input_args = prefix + "_%06d.png"
    input_args = "%06d.png"
    input_args = os.path.join(rgb_dir, input_args)

    os.system(f"ffmpeg -y -r {fps} -i " + input_args + f" -b:a 129k -crf 10 {video_save_path} >/dev/null 2>&1")


def generate_video_from_rgbs_batch(root_dir: str, fps: float):
    audio_video_dir = os.path.join(root_dir, "audio_video")
    seq_list = sorted(os.listdir(audio_video_dir))
    seq_dir_list = [os.path.join(audio_video_dir, x) for x in seq_list if not x.endswith(".pkl")]

    for seq_i, seq_dir in enumerate(seq_dir_list):
        seq_vid_path = os.path.join(seq_dir, "output.mp4")
        print(f"Generating video {seq_i}: {seq_vid_path}")
        if os.path.exists(seq_vid_path):
            continue
        rgb_dir = os.path.join(seq_dir, "rgb")
        generate_video_from_rgbs(seq_vid_path, rgb_dir, fps)


def select_audio_frames(
    rgb_dir,
    meta_save_dir: str,
    poses: np.ndarray,
    avoid_pos: np.ndarray,
    interval_s: float = 1,
    min_dist_between_sound: float = 1,
    fps: float = 30,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    rgb_path_list = [os.path.join(rgb_dir, x) for x in sorted(os.listdir(rgb_dir))]
    audio_start_frame = -1
    audio_end_frame = -1
    start_end_pairs = []

    # pose_path_list = [os.path.join(pose_dir, x) for x in sorted(os.listdir(pose_dir))]
    seq_pos = poses[:, :3]
    # for pose_i, pose_path in enumerate(pose_path_list):
    #     pos, rot = load_pose(pose_path)
    #     seq_pos[pose_i, :] = pos.flatten()
    l, r = 0, 0
    while l < seq_pos.shape[0]:
        r = l + 5 * fps
        print(l, r)
        group_pos = seq_pos[l:r]
        dists = cdist(group_pos, avoid_pos)
        if np.min(dists) > 2.0:
            r = np.min([r, seq_pos.shape[0] - 1])
            start_end_pairs.append((l, r))
            l = r + fps
            group_pos_unique = np.unique(group_pos, axis=0)
            avoid_pos = np.concatenate([avoid_pos, group_pos_unique], axis=0)
            continue
        group_min_dists = np.min(dists, axis=1)
        ids = np.where(group_min_dists <= 3.0)[0] + l
        max_id = np.max(ids)
        l = max_id + 1

    with open(meta_save_dir, "w") as f:
        sep = ""
        for pair_i, (start, end) in enumerate(start_end_pairs):
            f.write(f"{sep}{start},{end}")
            sep = "\n"
    return start_end_pairs, avoid_pos


def select_audio_frames_batch(
    root_dir: str,
    interval_s: float = 1,
    min_dist_between_sound: float = 2,
    fps: float = 30,
    overwrite: bool = True,
):
    audio_video_dir = os.path.join(root_dir, "audio_video")
    seq_list = sorted(os.listdir(audio_video_dir))
    seq_dir_list = [os.path.join(audio_video_dir, x) for x in seq_list if not x.endswith(".pkl")]
    avoid_pos = np.array([[np.inf, np.inf, np.inf]], dtype=np.float32)

    for seq_i, seq_dir in enumerate(seq_dir_list):
        print(f"sequence {seq_i}")
        pose_path = os.path.join(seq_dir, "poses.txt")
        poses = np.loadtxt(pose_path)
        rgb_dir = os.path.join(seq_dir, "rgb")
        meta_path = os.path.join(seq_dir, "meta.txt")
        if os.path.exists(meta_path) and not overwrite:
            continue
        _, avoid_pos = select_audio_frames(
            rgb_dir,
            meta_path,
            poses,
            avoid_pos,
            interval_s=interval_s,
            min_dist_between_sound=min_dist_between_sound,
            fps=fps,
        )


def assign_sound_to_video(
    audio_dir: str,
    esc_meta_path: str,
    seq_dir: str,
    considered_categories: List[str],
    difficulty_level: str,
    fps: float,
    pos_list: np.ndarray = None,
    occupied_positions: List[np.ndarray] = None,
    unassigned_categories: Set[str] = None,
):
    video_path = os.path.join(seq_dir, "output.mp4")
    slot_meta_path = os.path.join(seq_dir, "meta.txt")
    output_path = os.path.join(seq_dir, f"output_with_audio_{difficulty_level}.mp4")
    os.system(f"rm {output_path}")
    range_and_audio_meta_path = os.path.join(seq_dir, f"range_and_audio_meta_{difficulty_level}.txt")

    # get the frame ranges
    frame_ranges_list = []
    with open(slot_meta_path, "r") as f:
        for line in f:
            frame_ranges_list.append([int(x) for x in line.split(",")])
    time_ranges = []
    frame_ranges = []
    pos_groups = []
    for pair_i, start_end_pair in enumerate(frame_ranges_list):
        pos_group = []
        for fi in range(start_end_pair[0], start_end_pair[1]):
            pos = pos_list[fi]
            pos_group.append(pos)
        pos_groups.append(pos_group)
        pos_group = np.array(pos_group).reshape((-1, 3))
        if occupied_positions is not None:
            occupied_pos_group = np.array(occupied_positions).reshape((-1, 3))
            # dist_mat = cdist(occupied_pos_group, pos_group)
            # min_dist = np.min(dist_mat)

        frame_ranges.append(start_end_pair)
        frame_range = start_end_pair[1] - start_end_pair[0]
        start_time = start_end_pair[0] / fps
        end_time = start_end_pair[1] / fps
        time_range = float(frame_range) / fps
        time_ranges.append((start_time, end_time, time_range))

    audio_meta = ESC50Meta(esc_meta_path, audio_dir)
    cat2path_dict = audio_meta.get_category_name_to_path_dict()
    print("fold 1 categories number: ", len(set(cat2path_dict.keys())))
    # selected_cat_list = np.random.randint(0, len(catid2path_dict.keys()), len(time_ranges)).tolist()
    if unassigned_categories is not None:
        unassigned_cat_list = sorted(list(unassigned_categories))
        print("considered unassigned categories: ", unassigned_cat_list)
        np.random.shuffle(unassigned_cat_list)
        if len(unassigned_cat_list) < len(time_ranges):
            time_ranges = time_ranges[: len(unassigned_cat_list)]
        selected_cat_names_list = [unassigned_cat_list[i] for i in range(len(time_ranges))]
    else:
        print("considered categories: ", considered_categories)
        selected_catid_list = np.random.choice(len(considered_categories), len(time_ranges)).tolist()
        print(selected_catid_list)
        selected_cat_names_list = [considered_categories[i] for i in selected_catid_list]
    audio_paths_list = [np.random.choice(cat2path_dict[cat], 1)[0] for cat in selected_cat_names_list]

    mix_command = "ffmpeg -y"
    arg = ' -filter_complex "'
    tmp_alias = ""
    for range_i, (time_range, audio_path, cat_name) in enumerate(
        zip(time_ranges, audio_paths_list, selected_cat_names_list)
    ):
        st, et, t_range = time_range
        # command += f" -t {t_range} -itsoffset {st} -i {audio_path}"
        mix_command += f" -t {t_range} -itsoffset {st} -i {audio_path}"
        arg += f"[{range_i}:a] adelay={int(st * 1000)}|{int(st * 1000)} [a{range_i}];"
        tmp_alias += f"[a{range_i}]"

    arg += f' {tmp_alias} amix=inputs={len(time_ranges)}:duration=longest"'
    pardir = str(pathlib.Path(output_path).parent.absolute())
    print("########################################################")
    print(pardir)
    tmp_audio_path = os.path.join(pardir, f"output_{difficulty_level}.wav")
    os.system(f"rm {tmp_audio_path}")
    mix_command += f" {arg} {tmp_audio_path} >/dev/null 2>&1"
    command = f"ffmpeg -y -i {video_path} -i {tmp_audio_path} -async 1 {output_path} >/dev/null 2>&1"
    print(mix_command)
    os.system(mix_command)
    print("**************************************************************")
    print(command)
    os.system(command)

    # write meta
    with open(range_and_audio_meta_path, "w") as f:
        sep = ""
        for range_i, (frame_range, time_range, audio_path, cat_name) in enumerate(
            zip(
                frame_ranges_list,
                time_ranges,
                audio_paths_list,
                selected_cat_names_list,
            )
        ):
            f.write(f"{sep}{frame_range[0]},{frame_range[1]},{time_range[0]},{time_range[1]},{cat_name},{audio_path}")
            sep = "\n"

    return time_ranges, frame_ranges, selected_cat_names_list, audio_paths_list


def get_level_categories(difficulty_level: str, sound_config: DictConfig):
    major2categories_dict = sound_config["major_categories"]
    level_major_categories = sound_config["difficulty"][difficulty_level]
    level_categories = []
    for major_cat in level_major_categories:
        level_categories.extend([x.replace("_", " ") for x in major2categories_dict[major_cat]])
    return sorted(level_categories)


def setup_audio_paths(root_dir: str) -> Tuple[str, List[str]]:
    audio_video_dir = os.path.join(root_dir, "audio_video")
    seq_list = os.listdir(audio_video_dir)
    seq_dir_list = sorted([os.path.join(audio_video_dir, x) for x in seq_list if not x.endswith(".pkl")])
    return audio_video_dir, seq_dir_list


def assign_sound_to_video_batch(
    audio_dir: str,
    esc_meta_path: str,
    root_dir: str,
    difficulty_level: str,
    sound_config: DictConfig,
    fps: float,
    seq_num: int = None,
):
    audio_video_dir, seq_dir_list = setup_audio_paths(root_dir)
    level_categories = get_level_categories(difficulty_level, sound_config)
    if seq_num is not None:
        seq_dir_list = seq_dir_list[:seq_num]

    unassigned_categories = set(level_categories)
    print(unassigned_categories)
    assigned_categories = set()
    occupied_positions = [np.ones(3) * np.inf]
    print(unassigned_categories)
    for seq_i, seq_dir in enumerate(seq_dir_list):
        print(f"assigning audio to video for sequence {seq_i}")
        # if len(unassigned_categories) == 0:
        #     break
        # pose_dir = os.path.join(seq_dir, "pose")
        # pose_list = [os.path.join(pose_dir, x) for x in sorted(os.listdir(pose_dir))]
        # pos_list = [load_pose(x)[0].flatten() for x in pose_list]
        pose_path = os.path.join(seq_dir, "poses.txt")
        pose_list = np.loadtxt(pose_path)
        pos_list = pose_list[:, :3]
        (
            time_ranges,
            frame_ranges,
            selected_cat_names_list,
            audio_paths_list,
        ) = assign_sound_to_video(
            audio_dir,
            esc_meta_path,
            seq_dir,
            level_categories,
            difficulty_level,
            fps,
            pos_list,
            occupied_positions,
            unassigned_categories,
        )

        # record occupied positions
        for range_i, frame_range in enumerate(frame_ranges):
            sf, ef = frame_range
            for fi in range(sf, ef):
                occupied_positions.append(pos_list[fi])

        # compute unassigned categories
        assigned_categories = assigned_categories.union(set(selected_cat_names_list))
        unassigned_categories = unassigned_categories.difference(assigned_categories)


def get_pos_group(
    seq_dir: str,
    difficulty_level: str,
    fps: float,
    pos_list: List[np.ndarray] = None,
):
    slot_meta_path = os.path.join(seq_dir, "meta.txt")

    # get the frame ranges
    frame_ranges_list = []
    with open(slot_meta_path, "r") as f:
        for line in f:
            frame_ranges_list.append([int(x) for x in line.split(",")])
    time_ranges = []
    frame_ranges = []
    pos_groups = []
    for pair_i, start_end_pair in enumerate(frame_ranges_list):
        pos_group = []
        for fi in range(start_end_pair[0], start_end_pair[1]):
            pos = pos_list[fi]
            pos_group.append(pos)
        pos_groups.append(pos_group)
        pos_group = np.array(pos_group).reshape((-1, 3))

        frame_ranges.append(start_end_pair)
        frame_range = start_end_pair[1] - start_end_pair[0]
        start_time = start_end_pair[0] / fps
        end_time = start_end_pair[1] / fps
        time_range = float(frame_range) / fps
        time_ranges.append((start_time, end_time, time_range))

    return pos_groups, frame_ranges, time_ranges


def assign_sound_to_video_manually(
    audio_dir: str,
    esc_meta_path: str,
    categories_list: List[str],
    seq_ids: List[int],
    frame_ranges: List[Tuple[int, int]],
    time_ranges: List[Tuple[float, float]],
    root_dir: str,
    difficulty_level: str,
    sound_config: DictConfig,
    fps: float,
    seq_num: int = None,
) -> None:
    audio_meta = ESC50Meta(esc_meta_path, audio_dir)
    cat2path_dict = audio_meta.get_category_name_to_path_dict()
    print("fold 1 categories number: ", len(set(cat2path_dict.keys())))
    # selected_cat_list = np.random.randint(0, len(catid2path_dict.keys()), len(time_ranges)).tolist()
    audio_paths_list = [np.random.choice(cat2path_dict[cat], 1)[0] for cat in categories_list]

    unique_ids = sorted(list(set(seq_ids)))
    seq_id_dict = defaultdict(list)
    for id, seq_id in enumerate(seq_ids):
        seq_id_dict[seq_id].append(id)

    for seq_id in seq_id_dict.keys():
        mix_command = "ffmpeg -y"
        arg = ' -filter_complex "'
        tmp_alias = ""
        video_path = os.path.join(root_dir, "audio_video", f"{seq_id:06}", "output.mp4")
        output_path = os.path.join(
            root_dir,
            "audio_video",
            f"{seq_id:06}",
            f"output_with_audio_manual_{difficulty_level}.mp4",
        )
        if os.path.exists(output_path):
            os.system(f"rm {output_path}")
        range_and_audio_meta_path = os.path.join(
            root_dir,
            "audio_video",
            f"{seq_id:06}",
            f"range_and_audio_manual_meta_{difficulty_level}.txt",
        )
        for range_i, id in enumerate(seq_id_dict[seq_id]):
            print("range i: ", range_i)
            time_range = time_ranges[id]
            audio_path = audio_paths_list[id]
            cat_name = categories_list[id]
            st, et, t_range = time_range
            # command += f" -t {t_range} -itsoffset {st} -i {audio_path}"
            mix_command += f" -t {t_range} -itsoffset {st} -i {audio_path}"
            arg += f"[{range_i}:a] adelay={int(st * 1000)}|{int(st * 1000)} [a{range_i}];"
            tmp_alias += f"[a{range_i}]"

        arg += f' {tmp_alias} amix=inputs={len(seq_id_dict[seq_id])}:duration=longest"'
        pardir = str(pathlib.Path(output_path).parent.absolute())
        print("########################################################")
        print(pardir)
        tmp_audio_path = os.path.join(pardir, f"output_manual_{difficulty_level}.wav")
        os.system(f"rm {tmp_audio_path}")
        mix_command += f" {arg} {tmp_audio_path} >/dev/null 2>&1"
        command = f"ffmpeg -y -i {video_path} -i {tmp_audio_path} -async 1 {output_path} >/dev/null 2>&1"
        print(mix_command)
        os.system(mix_command)
        print("**************************************************************")
        print(command)
        os.system(command)

        # write meta
        with open(range_and_audio_meta_path, "w") as f:
            sep = ""
            for range_i, id in enumerate(seq_id_dict[seq_id]):
                frame_range = frame_ranges[id]
                time_range = time_ranges[id]
                audio_path = audio_paths_list[id]
                cat_name = categories_list[id]
                f.write(
                    f"{sep}{frame_range[0]},{frame_range[1]},{time_range[0]},{time_range[1]},{cat_name},{audio_path}"
                )
                sep = "\n"


def assign_sound_to_video_manually_batch(
    audio_dir: str,
    esc_meta_path: str,
    root_dir: str,
    difficulty_level: str,
    sound_config: DictConfig,
    fps: float,
    seq_num: int = None,
):
    audio_video_dir, seq_dir_list = setup_audio_paths(root_dir)
    level_categories = get_level_categories(difficulty_level, sound_config)
    if seq_num is not None:
        seq_dir_list = seq_dir_list[:seq_num]

    dataloader = VLMapsDataloaderHabitat(root_dir, "gt")
    dataloader.visualize_gt()

    all_pos_groups = []
    all_frame_ranges = []
    all_time_ranges = []
    all_seq_ids = []
    for seq_i, seq_dir in enumerate(seq_dir_list):
        print(f"assigning audio to video for sequence {seq_i}")
        pose_dir = os.path.join(seq_dir, "pose")
        pose_list = [os.path.join(pose_dir, x) for x in sorted(os.listdir(pose_dir))]
        pos_list = [load_pose(x)[0].flatten() for x in pose_list]
        # get all segment poses and frame ranges
        (pos_groups, frame_ranges, time_ranges) = get_pos_group(
            seq_dir,
            difficulty_level,
            fps,
            pos_list,
        )
        all_pos_groups.extend(pos_groups)
        all_frame_ranges.extend(frame_ranges)
        all_time_ranges.extend(time_ranges)
        all_seq_ids.extend([seq_i] * len(frame_ranges))

    map = dataloader.get_obstacles_cropped_no_floor()
    map = np.tile(map[:, :, None] * 255, [1, 1, 3]).astype(np.uint8)

    all_cropped_pos_list = []
    for pos_group in all_pos_groups:
        cropped_pos_list = dataloader.convert_habitat_pos_list_to_cropped_map_pos_list(pos_group)
        all_cropped_pos_list.append(cropped_pos_list)

    assigner = CategoryAssigner(map, all_cropped_pos_list, level_categories)
    assigner.select_categories_for_positions()
    assigned_list = assigner.assigned_cat_list
    assign_sound_to_video_manually(
        audio_dir,
        esc_meta_path,
        assigned_list,
        all_seq_ids,
        all_frame_ranges,
        all_time_ranges,
        root_dir,
        difficulty_level,
        sound_config,
        fps,
        seq_num,
    )


def test_frame_num(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("video not opened for some reasons")
        raise

    fi = 0
    while cap.isOpened():
        # print(f"loading frame {fi}")
        ret, frame = cap.read()
        if ret:
            fi += 1
            continue
        break

    return fi


def generate_audio_video_full_pipeline(root_dir: str, esc50_audio_dir: str, esc50_meta_path: str, fps: float):
    select_audio_frames_batch(root_dir)
    generate_video_from_rgbs_batch(root_dir, fps)
    assign_sound_to_video_batch(esc50_audio_dir, esc50_meta_path, root_dir, fps)


def extract_audio_from_video(video_path: str, output_audio_path: str) -> np.ndarray:
    rm_audio = f"rm {output_audio_path}"
    os.system(rm_audio)
    command = f"ffmpeg -y -i {video_path} -vn {output_audio_path} >/dev/null 2>&1"
    os.system(command)


def segment_audio_with_silence(
    audio_path: str,
    silence_duration_s: float = 1,
    silence_thres: int = 0,
    sample_rate: int = 44100,
) -> Tuple[List[Tuple[float, float]], List[np.ndarray]]:
    if not os.path.exists(audio_path):
        return [], []
    audio, sr = lb.load(audio_path, sr=sample_rate)
    indices = np.where(audio > silence_thres)[0]
    silence_duration_samples = int(silence_duration_s * sample_rate)
    l = indices[0]
    r = indices[0]
    segment_time_ranges = []
    segment_tracks = []
    for id in indices:
        if id - r < silence_duration_samples:
            r = id
            continue

        time_range = lb.samples_to_time([l, r], sr=sample_rate)
        segment_time_ranges.append((time_range[0], time_range[1]))
        segment_tracks.append(audio[l:r])
        l = id
        r = id
        # print("change start at: ", lb.samples_to_time([l], sr=sample_rate)[0])
    time_range = lb.samples_to_time([l, r], sr=sample_rate)
    segment_time_ranges.append((time_range[0], time_range[1]))
    segment_tracks.append(audio[l:r])
    # print(segment_time_ranges)

    return segment_time_ranges, segment_tracks


def convert_time_ranges_to_frame_ranges(time_ranges: List[Tuple[float, float]], fps: float) -> List[Tuple[int, int]]:
    return [(int(s * fps), int(e * fps)) for (s, e) in time_ranges]


# call HLocLocalizer to localize video frames

# call encode_audio_batch in test_audioclip.py to generate features for audios


def create_audio_dictionary(audio_features: List[np.ndarray], locations: List[List[np.ndarray]]) -> Dict[int, Dict]:
    audio_dict = {}
    for id, (audio_feat, locations_list) in enumerate(zip(audio_features, locations)):
        print("id: ", id)
        print("audio_feat.shape: ", audio_feat.shape)
        print("audio_feat: ", audio_feat)
        audio_dict[id] = {"audio_features": audio_feat, "locations": locations_list}

    return audio_dict


def get_five_second_contexts_audio(audio: np.ndarray, times: List[float], sample_rate: int) -> np.ndarray:
    """Times is a list in seconds, return a list of 5-second audio clips."""
    subsets = []
    for time in times:
        if time - 2.5 > (audio.shape[-1] - 1) / sample_rate:
            continue
        start_end = [time - 2.5, time + 2.5]
        sample_indices = lb.time_to_samples(start_end, sr=sample_rate)
        subset = audio[sample_indices[0] : sample_indices[1]]
        tmp = np.zeros((1, 5 * sample_rate))
        tmp[0, : subset.shape[-1]] = subset
        subset = tmp

        subsets.append(subset)
    return np.array(subsets)


def encode_audio(track: np.ndarray, model: AudioCLIP, audio_transforms: ToTensor1D, sample_rate: int) -> np.ndarray:
    track = get_five_second_contexts_audio(track, [2.5], sample_rate)
    audio_batch = list()

    # compute spectrograms using trained audio-head (fbsp-layer of ESResNeXt)
    # thus, the actual time-frequency representation will be visualized
    spec = model.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
    spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
    pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()
    audio_batch.append((track, pow_spec))
    audio_batch = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio_batch])
    audio_feature: torch.Tensor
    ((audio_feature, _, _), _), _ = model(audio=audio_batch)
    return audio_feature.cpu().numpy()


def encode_audio_batch(
    tracks: List[np.ndarray],
    model: AudioCLIP,
    audio_transforms: tv.transforms.Compose,
    sample_rate: int,
):
    torch.set_grad_enabled(False)
    # model = model.eval()
    audio_batch = list()
    audio_features = list()
    single_sample_batch = False
    for audio_i, track in enumerate(tracks):
        track = get_five_second_contexts_audio(track, [2.5], sample_rate)
        spec = model.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
        spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
        pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()
        audio_batch.append((track, pow_spec))

        if audio_i % 10 == 9 or audio_i + 1 == len(tracks):
            if len(audio_batch) == 1:
                single_sample_batch = True
            if single_sample_batch:
                audio_batch.append(audio_batch[0])
            audio_batch = torch.stack([audio_transforms(t.reshape(1, -1)) for t, _ in audio_batch])
            audio_feature: torch.Tensor
            ((audio_feature, _, _), _), _ = model(audio=audio_batch)
            if single_sample_batch:
                audio_feature = audio_feature[0].reshape((1, -1))
            audio_features.append(audio_feature)
            audio_batch = list()
            single_sample_batch = False

    audio_features = torch.cat(audio_features, dim=0)
    if len(audio_features.shape) == 1:
        audio_features = audio_features.reshape((1, -1))

    # categories = ["toilet flush", "cock singing", "sheep", "brushing teeth", "hen", "car noise"]
    # texts = [[cat] for cat in categories]
    # ((_, _, text_features), _), _ = model(text=texts)
    # scale_audio_text = torch.clamp(model.logit_scale_at.exp(), min=1.0, max=100.0)
    # logits_audio_text = scale_audio_text * audio_features @ text_features.T
    # predicts = np.argmax(logits_audio_text.cpu().numpy(), axis=1)
    # confidence = logits_audio_text.softmax(dim=1)
    # print(predicts)
    # print(confidence)

    return audio_features.cpu().numpy()


def save_segment_tracks(segment_tracks: List[np.ndarray], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for track_i, track in enumerate(segment_tracks):
        filepath = os.path.join(save_dir, f"{track_i:06}.wav")
        soundfile.write(filepath, track, 44100)


def create_audioclip_audio_map(
    audio_dir: str, audio_names: List[str], pos_list: List[List[float]], sample_rate: int = 44100
):
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
    audio_paths = [os.path.join(audio_dir, x) for x in audio_names]
    segment_tracks = []
    for audio_i, audio_path in enumerate(audio_paths):
        audio, sr = lb.load(audio_path, sr=sample_rate)
        segment_tracks.append(audio)
    segment_tracks = [x.astype(np.float32) * 32768.0 for x in segment_tracks]
    audio_features = encode_audio_batch(segment_tracks, aclp, audio_transforms, sample_rate)
    return {"audio_features": audio_features, "positions_cropped": pos_list}


def create_audioclip_audio_text_scores(audio_features: np.ndarray, text_categories: List[str]):
    MODEL_FILENAME = "AudioCLIP-Partial-Training.pt"
    model_path = os.path.join("audioclip/assets/", MODEL_FILENAME)
    aclp = AudioCLIP(pretrained=model_path).eval()
    categories = [["the sound of " + x] for x in text_categories]

    audio_features = torch.from_numpy(audio_features)
    with torch.no_grad():
        ((_, _, text_features), _), _ = aclp(text=categories)
        scale_audio_text = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
        logits_audio_text = scale_audio_text * audio_features @ text_features.T
    return logits_audio_text.cpu().numpy()


def create_wav2clip_audio_map(
    audio_dir: str, audio_names: List[str], pos_list: List[List[float]], sample_rate: int = 16000
):
    audio_paths = [os.path.join(audio_dir, x) for x in audio_names]
    segment_tracks = []
    for audio_i, audio_path in enumerate(audio_paths):
        audio, sr = lb.load(audio_path, sr=sample_rate)
        segment_tracks.append(audio)
    wav2clip_model = wav2clip.get_model()
    audio_features = np.zeros((len(segment_tracks), 512))
    pbar = tqdm.tqdm(enumerate(segment_tracks), total=len(segment_tracks))
    for aud_i, audio in pbar:
        pbar.set_description(f"Computing embedding for audio {aud_i:06}: {id}, audio length: {audio.shape[-1] / sr}s")
        audio_feats = wav2clip.embed_audio(audio, wav2clip_model)
        audio_feats = torch.from_numpy(audio_feats)
        audio_feats /= audio_feats.norm(dim=-1, keepdim=True)
        audio_feats = np.float32(audio_feats)
        audio_features[aud_i, :] = audio_feats

    return {"audio_features": audio_features, "positions_cropped": pos_list}


def create_wav2clip_audio_text_scores(audio_features: np.ndarray, text_categories: List[str]):
    clip_model_b32, preprocess = clip.load("ViT-B/32")  # clip.available_models()
    clip_model_b32.cuda().eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    text_feats = get_text_feats(text_categories, clip_model_b32, clip_feat_dim)
    scores_list = audio_features @ text_feats.T
    return scores_list


def main():
    extract_audio_from_video(
        "/home/hcg/hcg/phd/projects/vln/data/clip_mapping/5LpN3gDmAk7_1/audio_video/000000/output_with_audio.mp4",
        "/tmp/output.wav",
    )


if __name__ == "__main__":
    main()
