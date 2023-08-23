# Copyright 2023 Shan Wu
# 
# * Playback with customized data
import configparser
import csv
import os
import os.path as osp
from pathlib import Path
from argparse import Namespace
import signal
from typing import List, Tuple, Union, Optional

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


class CustomSequence(Dataset):
    """
    CustomSequence (Delta) Dataset.
    """

    def __init__(self, root_dir: str, seq_name: Optional[str] = None, img_transform: Namespace = None,
                 include_original_img: bool = False) -> None:
        """
        Args:
            seq_name (string): Sequence to take
        """
        super().__init__()

        # .../Delta
        self._data_dir = Path(root_dir)
        if not self._data_dir.exists() or not self._data_dir.is_dir():
            raise ValueError(f'data_root_dir:{root_dir} does not exist.')

        # .../Delta/seq/
        if seq_name is None:
            seqs = [f.name for f in self._data_dir.iterdir() if f.is_dir()]
            idx = item_chooser([': '.join((str(i), d)) for i, d in enumerate(seqs)])
            seq_name = seqs[idx]
        self._seq_dir = self._data_dir.joinpath(seq_name)
        if not self._seq_dir.exists():
            raise ValueError(f'Sequence {seq_name} not found in data folder at {self._data_dir}.')

        # .../Delta/seq/src/
        self._src_dir = self._seq_dir.joinpath('src')
        if not self._src_dir.exists():
            raise ValueError(f'No source (src) directory in the sequence {seq_name}.')

        # .../Delta/seq/seqinfo.ini
        self._config_path = self._seq_dir.joinpath('seqinfo.ini')
        if not self._config_path.exists():
            raise ValueError(f'No configuration file (seqinfo.ini) in the sequence {seq_name}.')
        self.config = self._get_config()

        # .../Delta/seq/gt/
        self._gt_dir = self._seq_dir.joinpath('gt')
        if not self._gt_dir.exists():
            print(f'[WARNING]: No gt files in the sequence {seq_name}.')
            self.no_gt = True
        else:
            self.no_gt = False

        # make transforms
        self.transforms = Compose(make_coco_transforms('val', img_transform, overflow_boxes=True))

        # load dataset
        self._is_video = False
        self.data = self._sequence()

        # attributes
        self._seq_name = seq_name
        self._incl_orig = include_original_img

        # validate
        assert self.__len__() == self.seq_length, \
            f"Actual data length (frames) doesn't match the length specified in the config file (seqinfo.ini)."
        return

    def __len__(self) -> int:
        return len(self.data)

    def __str__(self) -> str:
        return f"{self._seq_name}"

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
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']

        if self._incl_orig:
            sample['orig_img'] = np.array(orig_img)
        return sample

    def _sequence(self) -> List[dict]:
        # determine src dir
        if self.config['imDir'] != 'src':
            print(
                f"[WARNING] Searching srouce files in the new directory ({self.config['imDir']}) based on configuration...")
            src_dir = self._seq_dir.joinpath(self.config['imDir'])
        else:
            src_dir = self._src_dir

        # determine src type
        boxes, visibility = self.get_track_boxes_and_visbility()
        if self.config['imExt'] in ['.png', '.jpg']:
            self._is_video = False
            ext = self.config['imExt']
            total = [
                {
                    'im_path': osp.join(src_dir, f"{i:06d}{ext}"),
                    'gt': boxes[i],
                    'vis': visibility[i]
                }
                for i in range(1, self.seq_length + 1)
            ]
        elif self.config['imExt'] in ['.mov', '.mp4', 'avi']:
            self._is_video = True
            video_file = self._get_vid_file(src_dir.as_posix(), ext=self.config['imExt'])
            frames, _, _ = read_video(video_file, output_format='TCHW')
            total = [
                {
                    'frame': frames[i - 1],
                    'im_path': video_file,
                    'gt': boxes[i],
                    'vis': visibility[i]
                }
                for i in range(1, self.seq_length + 1)
            ]
        else:
            raise ValueError('Cannot determine the type of data source.')

        return total

    def get_track_boxes_and_visbility(self) -> Tuple[dict, dict]:
        """ Load ground truth boxes and their visibility."""
        boxes = {}
        visibility = {}

        for i in range(1, self.seq_length + 1):
            boxes[i] = {}
            visibility[i] = {}

        gt_file = self._gt_dir.joinpath('gt.txt')
        if not osp.exists(gt_file):
            return boxes, visibility

        with open(gt_file, "r") as inf:
            reader = csv.reader(inf, delimiter=',')
            for row in reader:
                # Make pixel indexes 0-based, should already be 0-based (or not)
                x1 = int(row[2]) - 1
                y1 = int(row[3]) - 1
                # This -1 accounts for the width (width of 1 x1=x2)
                x2 = x1 + int(row[4]) - 1
                y2 = y1 + int(row[5]) - 1
                bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

                frame_id = int(row[0])
                track_id = int(row[1])

                boxes[frame_id][track_id] = bbox
                visibility[frame_id][track_id] = 1.0

        return boxes, visibility

    def load_results(self, results_dir: str) -> dict:
        results = {}
        if results_dir is None:
            return results

        file_path = osp.join(results_dir, self.results_file_name)

        if not os.path.isfile(file_path):
            return results

        with open(file_path, "r") as file:
            csv_reader = csv.reader(file, delimiter=',')

            for row in csv_reader:
                frame_id, track_id = int(row[0]) - 1, int(row[1]) - 1

                if track_id not in results:
                    results[track_id] = {}

                x1 = float(row[2]) - 1
                y1 = float(row[3]) - 1
                x2 = float(row[4]) - 1 + x1
                y2 = float(row[5]) - 1 + y1

                results[track_id][frame_id] = {}
                results[track_id][frame_id]['bbox'] = [x1, y1, x2, y2]
                results[track_id][frame_id]['score'] = 1.0

        return results

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

        result_file_path = osp.join(output_dir, self.results_file_name)

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

    @staticmethod
    def _get_vid_file(src_dir: str, ext: str = '.mp4', top_p: int = 1) -> Union[str, List[str]]:
        # find videos
        video_file = []
        for filename in sorted(os.listdir(src_dir)):
            extension = os.path.splitext(filename)[1]
            if extension == ext:
                video_file.append(osp.join(src_dir, filename))

        if len(video_file) < 1:
            raise ValueError(f'No video file found in the source folder. Tried to search {ext} files.')

        if top_p <= 1:
            return video_file[0]
        else:
            return video_file[:top_p]

    def _get_config(self):
        config = configparser.ConfigParser()
        config.read(self._config_path)
        return config['Sequence']

    @property
    def seq_length(self) -> int:
        """ Return sequence length, i.e, number of frames. """
        return int(self.config['seqLength'])

    @property
    def im_width(self):
        return int(self.config['imWidth'])

    @property
    def im_height(self):
        return int(self.config['imHeight'])

    @property
    def results_file_name(self) -> str:
        """ Generate file name of results file. """
        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"
        return f"{self}.txt"
