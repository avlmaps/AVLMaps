import os
from pathlib import Path
import pickle
import numpy as np
import torch
import torchvision as tv
import librosa as lb
from omegaconf import DictConfig
from typing import Dict, List, Tuple

from avlmaps.utils.audio_utils import get_level_categories, encode_audio_batch, encode_audio
from avlmaps.utils.index_utils import find_similar_category_id
from avlmaps.audioclip.model.audioclip import AudioCLIP
from avlmaps.audioclip.utils.transforms import ToTensor1D, RandomCrop, RandomPadding
from avlmaps.utils.audio_mapping_utils import create_audio_map_batch, create_audio_map_statistics


class SoundMap:
    def __init__(
        self,
        avlmaps_data_dir: str,
        sound_config: DictConfig,
        sound_data_collect_config: DictConfig,
        is_ambiguous: bool = False,
        is_real: bool = False,
    ):
        self.avlmaps_data_dir = avlmaps_data_dir
        self.difficulty_level = sound_data_collect_config["difficulty"]
        self.sound_config = sound_config
        self.manual_str = "_manual" if is_ambiguous else ""
        self.is_real = is_real
        self.sound_categories = self._load_sound_categories()
        # self.audio_database = self.load_sound_map()
        self.sound_data_collet_config = sound_data_collect_config
        self._init_audioclip()

    def _init_audioclip(self):
        MODEL_FILENAME = "AudioCLIP-Partial-Training.pt"
        model_path = Path(__file__).parent.parent / "audioclip" / "assets" / MODEL_FILENAME
        self.aclp = AudioCLIP(pretrained=model_path.as_posix()).eval()
        self.audio_transforms = tv.transforms.Compose(
            [
                ToTensor1D(),
                RandomPadding(out_len=220500, train=False),
                RandomCrop(out_len=220500, train=False),
            ]
        )

    def _load_sound_categories(self):
        return get_level_categories(self.difficulty_level, self.sound_config)

    def create_sound_map(self, data_dir: str):
        data_dir = Path(data_dir)
        create_audio_map_batch(
            data_dir.as_posix(),
            aclp=self.aclp,
            audio_transforms=self.audio_transforms,
            sample_rate=self.sound_data_collet_config["sample_rate"],
            silence_duration_s=self.sound_data_collet_config["silence_duration_s"],
            silence_thres=self.sound_data_collet_config["silence_threshold"],
            fps=self.sound_data_collet_config["fps"],
            difficulty_level=self.sound_data_collet_config["difficulty"],
            manual_mode=False,
            seq_num=self.sound_data_collet_config["considered_seq_num_per_scene"],
        )
        create_audio_map_statistics(
            data_dir,
            difficulty_level=self.sound_data_collet_config["difficulty"],
            manual_mode=False,
            seq_num=self.sound_data_collet_config["considered_seq_num_per_scene"],
        )

    def load_sound_map(self, data_dir: str):
        filename = "audio_data.pkl" if self.is_real else f"audio_data{self.manual_str}_{self.difficulty_level}.pkl"

        data_dir = Path(data_dir)

        sound_map_path = data_dir / "audio_video" / filename

        with open(sound_map_path, "rb") as f:
            # audio_database contains a list of ids as keys.
            # each id is mapped to a dictionary with "audio_features" and "locations" as keys
            self.audio_database = pickle.load(f)
            return self.audio_database

    def get_all_audio_features_and_locations(
        self,
    ) -> Tuple[np.ndarray, List[List[np.ndarray]]]:
        audio_features = []
        feature_locations = []
        for id in range(len(self.audio_database.keys())):
            audio_feat = self.audio_database[id]["audio_features"]
            locations = self.audio_database[id]["locations"]
            audio_features.append(audio_feat)
            feature_locations.append(locations)
        audio_features = np.stack(audio_features, axis=0)
        return audio_features, feature_locations

    def _get_pos_with_classification(self, logits_audio_text: torch.Tensor, categories: List[str]):
        predicts = np.argmax(logits_audio_text.cpu().numpy(), axis=1)

    def get_pos(self, name: str):
        audio_features, feature_locations = self.get_all_audio_features_and_locations()
        audio_features = torch.from_numpy(audio_features)
        texts = [[cat] for cat in self.sound_categories]
        with torch.no_grad():
            ((_, _, text_features), _), _ = self.aclp(text=texts)
            scale_audio_text = torch.clamp(self.aclp.logit_scale_at.exp(), min=1.0, max=100.0)
            logits_audio_text = scale_audio_text * audio_features @ text_features.T
            confidence = logits_audio_text.softmax(dim=1)
            predicts = np.argmax(logits_audio_text.cpu().numpy(), axis=1)
            retrieval_confidence = logits_audio_text.softmax(dim=0)
            retrievals = np.argmax(logits_audio_text.cpu().numpy(), axis=0)
            print((retrieval_confidence * 100).cpu().numpy().astype(int))
            print("retrievals: ", retrievals)

        cat_id = find_similar_category_id(name, self.sound_categories)
        retrieval_id = retrievals[cat_id]
        locations = feature_locations[retrieval_id]
        return locations

    def get_pos_with_audio(self, audio_path: str, sample_rate: int):
        if not os.path.exists(audio_path):
            return [], []
        audio, sr = lb.load(audio_path, sr=sample_rate)
        audio = audio.astype(np.float32) * 32768.0
        query_audio_features = encode_audio(audio, self.aclp, self.audio_transforms, sample_rate).reshape((1, -1))
        audio_features, feature_locations = self.get_all_audio_features_and_locations()
        sim_mat = audio_features @ query_audio_features.T

        retrieval_id = np.argmax(sim_mat)
        locations = feature_locations[retrieval_id]
        return locations

    def get_distribution_and_locations(self, name: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        audio_features, feature_locations = self.get_all_audio_features_and_locations()
        audio_features = torch.from_numpy(audio_features)
        texts = [[cat] for cat in self.sound_categories]
        with torch.no_grad():
            ((_, _, text_features), _), _ = self.aclp(text=texts)
            scale_audio_text = torch.clamp(self.aclp.logit_scale_at.exp(), min=1.0, max=100.0)
            logits_audio_text = scale_audio_text * audio_features @ text_features.T
            confidence = logits_audio_text.softmax(dim=1)
            predicts = np.argmax(logits_audio_text.cpu().numpy(), axis=1)
            retrieval_confidence = logits_audio_text.softmax(dim=0)
            retrievals = np.argmax(logits_audio_text.cpu().numpy(), axis=0)
            # print((retrieval_confidence * 100).cpu().numpy().astype(int))
            # print("retrievals: ", retrievals)

        cat_id = find_similar_category_id(name, self.sound_categories)
        probabilities = logits_audio_text.cpu().numpy()[:, cat_id]
        probabilities = (probabilities - np.min(probabilities)) / (np.max(probabilities) - np.min(probabilities))
        return probabilities, feature_locations
