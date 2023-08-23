# Copyright 2023 Shan Wu
# Modification by Shan
# * Added feature to demonstrate with videos
# ---
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT17 sequence dataset.
"""
import configparser
import csv
import os
from pathlib import Path
import os.path as osp
from argparse import Namespace
from typing import List
import signal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_video
import torchvision.transforms.functional as F

from ..coco import make_coco_transforms
from ..transforms import Compose


def item_chooser(menu: List[str], timeout: int = 15) -> int:
    def timeout_handler(signum, frame):
        raise TimeoutError('Timeout. The first index will be used.')

    # compose menu content
    valid_idx = list(range(len(menu)))
    menu_str = '\n'.join(menu)
    print(f'Find following video file: \n'
          f'------------------------------\n'
          f'{menu_str}\n'
          f'------------------------------')

    # set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        while True:
            idx = input("> Select a video file's idx to demonstrate: ").strip()
            if idx.isdigit():
                idx = int(idx)
                if idx not in valid_idx:
                    print('Index out of range.')
                else:
                    # valid input
                    signal.alarm(0)
                    break
            else:
                print('Invalid input.')
    except TimeoutError as e:
        idx = 0

    print(f'\nSelected item: {menu[idx]}')
    return idx


class DemoSequence(Dataset):
    """DemoSequence (MOT17) Dataset.
    """

    def __init__(self, root_dir: str = 'data', img_transform: Namespace = None,
                 include_original_img: bool = False) -> None:
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons
                                   above which they are selected
        """
        super().__init__()
        self._incl_orig = include_original_img

        self._data_dir = Path(root_dir)
        assert self._data_dir.is_dir(), f'data_root_dir:{root_dir} does not exist.'

        self.transforms = Compose(make_coco_transforms('val', img_transform, overflow_boxes=True))

        self._is_video = False
        self.data = self._sequence()
        self.no_gt = True

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return self._data_dir.name

    def __getitem__(self, idx: int) -> dict:
        """Return the ith image converted to blob"""
        if not self._is_video:
            data = self.data[idx]
            orig_img = Image.open(data['im_path']).convert("RGB")
            width_orig, height_orig = orig_img.size
        else:
            data = self.data[idx]
            orig_img = F.to_pil_image(data['frame'])
            width_orig, height_orig = orig_img.size

        img, _ = self.transforms(orig_img)
        width, height = img.size(2), img.size(1)

        sample = {}
        sample['img'] = img
        sample['img_path'] = data['im_path']
        sample['dets'] = torch.tensor([])
        sample['orig_size'] = torch.as_tensor([int(height_orig), int(width_orig)])
        sample['size'] = torch.as_tensor([int(height), int(width)])

        if self._incl_orig:
            sample['orig_img'] = np.array(orig_img)
        return sample

    def _sequence(self) -> List[dict]:
        total = []
        for filename in sorted(os.listdir(self._data_dir)):
            extension = os.path.splitext(filename)[1]
            if extension in ['.png', '.jpg']:
                total.append({'im_path': osp.join(self._data_dir, filename)})

        if len(total) == 0:
            self._is_video = True
            print(f'No image found in the directory: {self._data_dir}. Try to search videos...')

            # find videos
            video_file = []
            for filename in sorted(os.listdir(self._data_dir)):
                extension = os.path.splitext(filename)[1]
                if extension in ['.mov', '.mp4', 'avi']:
                    video_file.append(osp.join(self._data_dir, filename))

            if len(video_file) == 0:
                raise FileNotFoundError(f"Cannot find a video file in the folder: {self._data_dir}. "
                                        f"Make sure your video file's extension is one of {['.mov', '.mp4', 'avi']}.")

            list_of_files = [': '.join((str(i), f)) for i, f in enumerate(video_file)]
            idx = item_chooser(list_of_files)
            print('Loading video file into memory...')
            frames, _, _ = read_video(video_file[idx], output_format='TCHW')
            for i, frame in enumerate(frames):
                total.append({'frame': frame, 'im_path': video_file, 'frame_id': i})
        else:
            print(f'Found {len(total)} images in {self._data_dir}.')

        return total

    def load_results(self, results_dir: str) -> dict:
        return {}

    def write_results(self, results: dict, output_dir: str) -> None:
        """Write the tracks in the format for MOT16/MOT17 sumbission

        results: dictionary with 1 dictionary for every track with
                 {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result_file_path = osp.join(output_dir, self._data_dir.name)

        with open(result_file_path, "w") as r_file:
            writer = csv.writer(r_file, delimiter=',')

            for i, track in results.items():
                for frame, data in track.items():
                    x1 = data['bbox'][0]
                    y1 = data['bbox'][1]
                    x2 = data['bbox'][2]
                    y2 = data['bbox'][3]

                    writer.writerow([
                        frame + 1,
                        i + 1,
                        x1 + 1,
                        y1 + 1,
                        x2 - x1 + 1,
                        y2 - y1 + 1,
                        -1, -1, -1, -1])
