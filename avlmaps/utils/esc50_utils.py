import os
import csv
from collections import defaultdict

from typing import Tuple, Dict, List


class ESC50Meta:
    def __init__(self, meta_path: str, audio_dir: str):
        self.audio_dir = audio_dir
        self.meta = []
        with open(meta_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row_i, row in enumerate(reader):
                self.meta.append(row)

        self.cat_list = self._get_category_names()

    def get_category_id_to_path_dict(self, tar_fold: int = 1) -> Dict[int, List[str]]:
        cat2path_dict = defaultdict(list)
        for row_i, row in enumerate(self.meta):
            filename = row[0]
            fold = int(row[1])
            cat = int(row[2])
            cat_name = row[3].replace('_', ' ')

            if fold == tar_fold:
                filepath = os.path.join(self.audio_dir, filename)
                cat2path_dict[cat].append(filepath)
        for cat in cat2path_dict.keys():
            cat2path_dict[cat].sort()
        return cat2path_dict

    def get_category_name_to_path_dict(self, tar_fold: int = 1) -> Dict[str, List[str]]:
        cat2path_dict = defaultdict(list)
        for row_i, row in enumerate(self.meta):
            filename = row[0]
            fold = int(row[1])
            cat = int(row[2])
            cat_name = row[3].replace('_', ' ')

            if fold == tar_fold:
                filepath = os.path.join(self.audio_dir, filename)
                cat2path_dict[cat_name].append(filepath)
        for cat in cat2path_dict.keys():
            cat2path_dict[cat].sort()
        return cat2path_dict

    def _get_category_names(self) -> List[str]:
        catid2cat_name_dict = dict()
        for row_i, row in enumerate(self.meta):
            catid = int(row[2])
            cat_name = row[3].replace('_', ' ')
            catid2cat_name_dict[catid] = cat_name
        cat_list = [""] * len(catid2cat_name_dict.keys())
        for id in catid2cat_name_dict.keys():
            cat_list[id] = catid2cat_name_dict[id]
        return cat_list

    def get_category_names(self) -> List[str]:
        return self.cat_list
