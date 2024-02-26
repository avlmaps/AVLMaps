import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set

@dataclass
class CategoryAssigner:
    map: np.ndarray
    all_pos_list: List[List[np.ndarray]]
    categories: List[str]

    def select_categories_for_positions(self):
        self.map = self._show_all_pos()
        # self.ax = plt.gca()

        print("all pos number: ", len(self.all_pos_list))
        self.assigned_cat_list = []
        for pos_group_i, cropped_pos_list in enumerate(self.all_pos_list):
            self.fig, self.ax = plt.subplots()
            self.fig.set_size_inches(15, 15)
            self.fig.subplots_adjust(right=0.8)
            self._generate_button_ids(self.categories)
            mask = self.map.copy()
            # put assigned category on the map
            for assigned_i, assigned_category in enumerate(self.assigned_cat_list):
                assigned_pos_list = self.all_pos_list[assigned_i]
                pos = assigned_pos_list[0]
                cv2.putText(
                    mask,
                    f"{assigned_category}",
                    (int(pos[1]), int(pos[0])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

            # label the currently to be assigned pos with red
            for pose_i, pos in enumerate(cropped_pos_list):
                # pos = int(pos[0]), int(pos[1])
                print(pos)
                cv2.circle(mask, (int(pos[1]), int(pos[0])), 3, (0, 0, 255), 2)
            
            mask = Image.fromarray(mask)
            seg_gt = mask.convert("RGBA")
            self.ax.axis("off")
            self.ax.imshow(seg_gt)
            plt.show()

    def get_assigned_categories(self):
        return self.assigned_cat_list


    def _generate_button_ids(self, categories):
        self.ids_list = []
        self.ax_list = []
        self.button_list = []
        @dataclass
        class ButtonId:
            id: int
            cat: str
            assigned_cat_list: List[str]

            def on_click(self, event):
                print(self.id)
                print(self.cat)
                self.assigned_cat_list.append(self.cat)
                print(self.assigned_cat_list)
                plt.close()
        
        height = 0.05
        if len(categories) * height > 1.0:
            height = 1.0 / len(categories)
        for cat_i, cat in enumerate(categories):
            self.ids_list.append(ButtonId(cat_i, cat, self.assigned_cat_list))
            self.ax_list.append(self.fig.add_axes([0.8, 0.95 - cat_i * height, 0.15, height / 2.]))
            btn = Button(self.ax_list[-1], cat)
            # id_list.append(ButtonId(cat_i, cat))
            btn.on_clicked(self.ids_list[cat_i].on_click)
            self.button_list.append(btn)

    def _show_all_pos(self):
        for pose_list_i, pos_list in enumerate(self.all_pos_list):
            for pos_i, pos in enumerate(pos_list):
                cv2.circle(self.map, (int(pos[1]), int(pos[0])), 3, (0, 255, 0), 2)
        return self.map