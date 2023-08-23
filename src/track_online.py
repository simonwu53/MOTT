# Copyright 2023 Shan Wu

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
import time
from os import path as osp
from typing import Tuple, Optional

import numpy as np
import sacred
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader
import cv2

from trackformer.datasets.tracking import TrackDatasetFactory
from trackformer.models import initialize_model
from trackformer.models.tracker import Tracker
from trackformer.util.colormap import colormap
from trackformer.util.mask_utils import mask_overlay

ex = sacred.Experiment('track_online')
ex.add_config('cfgs/track_online.yaml')


class Player:
    def __init__(self, win_size: Tuple[int, int] = (1333, 800), win_name: str = "MOT_Player",
                 model: Optional = None, output_dir: Optional = None,
                 _delay: int = 1, _log: Optional = None, _debug: bool = False):
        self.width, self.height = win_size
        self.delay = _delay
        self.logger = _log
        self.debug = _debug
        self.colors = colormap(rgb=True)
        self.win_name = win_name

        self.model = model
        self.output_dir = output_dir

        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win_name, self.width, self.height)

        if output_dir is not None:
            fourcc = cv2.VideoWriter_fourcc(*'HEVC')
            fps = 30
            output_path = osp.join(output_dir, f'{self.win_name}.mp4')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, win_size)
            self.logger(f"[Player] Output video to path: {output_path}")
        else:
            self.video_writer = None

        if self.logger:
            self.logger(f"[Player] Player initialized: {self.win_name}")
        return

    def __del__(self):
        if self.logger:
            self.logger("[Player] Clean ups.")
        cv2.destroyAllWindows()
        if self.video_writer is not None:
            self.video_writer.release()
        if self.logger:
            self.logger("[Player] Destructed.")
        return

    def step(self, frame_data, orig_img):
        """
        frame_data: Dict,
            mandatory keys:
                'img': Tensor(1, c, h, w), after "transform"
                'orig_size': Tensor(2),
                'dets': Tensor(N, 4) or [[]],
            optional keys:
                'size': Tensor(2)
        orig_img: Numpy, (H, W, 3), RGB
        """
        assert self.model is not None, "No Tracking Model Loaded."
        with torch.no_grad():
            # model forwarding
            self.model.step(frame_data)
            # get tracking results
            tracks = self.model.tracks
            # visualization
            ret = self.interpret(orig_img, tracks)
        return ret

    def interpret(self, img, tracks):
        # render frame
        rendered = self.render_tracks(img, tracks)

        # show img
        frame = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.win_name, frame)
        # save video
        if self.output_dir is not None:
            self.video_writer.write(frame)

        if cv2.waitKey(self.delay) & 0xFF == ord('q'):
            return 0
        return 1

    def interpret_results(self, frame_id, img, results):
        # find tracklets in current frame
        tracks = []
        for track_id, track_data in results.items():
            if frame_id in track_data.keys():
                track_data[frame_id]["id"] = track_id
                tracks.append(track_data[frame_id])

        # render frame
        ret = self.interpret(img, tracks)
        return ret

    def render_tracks(self, img, tracks):
        for t in tracks:
            if isinstance(t, dict):
                tid = t["id"]
                bbox = list(map(int, t["bbox"]))
                score = t["score"]
                mask = t.get("mask", None)
            else:
                tid = t.id
                bbox = list(map(int, t.pos.detach().cpu().numpy()))
                score = t.score.item()
                mask = t.mask.cpu().numpy() if t.mask is not None else None

            # draw mask
            if mask is not None:
                img = mask_overlay(img, mask, color=self.colors[tid % 79].tolist())
            # draw bounding box
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          self.colors[tid % 79].tolist(), thickness=2)
            # put text
            cv2.putText(img, f"{tid}", (bbox[0], bbox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        self.colors[tid % 79].tolist(), 2)
            if self.debug:
                cv2.putText(img, f"{score:.2f}",
                            (bbox[0] + (bbox[2] - bbox[0]) // 2, bbox[1] + (bbox[3] - bbox[1]) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            self.colors[tid % 79].tolist(), 2)
        return img


@ex.automain
def main(seed, output_dir, verbose, obj_detect_checkpoint_file, dataset_name,
         data_root_dir, load_results_dir, frame_range, tracker_cfg, checkpoint_version,
         _config, _log, _run):
    # print config
    sacred.commands.print_config(_run)

    # set all seeds
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    # make output dir
    if output_dir is not None:
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
        # dump config
        yaml.dump(
            _config,
            open(osp.join(output_dir, 'track.yaml'), 'w'),
            default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################
    mott, mott_post, img_transform = initialize_model(obj_detect_checkpoint_file, _logger=_log,
                                                      ckpt_version=checkpoint_version)

    # logger
    track_logger = None
    if verbose:
        track_logger = _log.info

    # init tracker with detector
    tracker = Tracker(mott, mott_post, tracker_cfg, False, track_logger, verbose)

    # load dataset
    dataset = TrackDatasetFactory(
        dataset_name, root_dir=data_root_dir, img_transform=img_transform, include_original_img=True)

    ##########################
    #     Start Tracking     #
    ##########################
    time_total = 0
    num_frames = 0

    # iterate data sequences
    for seq in dataset:
        tracker.reset()
        _log.info(f"------------------")
        _log.info(f"TRACK SEQ: {seq}")

        # init video player
        if hasattr(seq, 'im_width') and hasattr(seq, 'im_height'):
            win_size = (seq.im_width, seq.im_height)
        else:
            win_size = (1333, 800)
            _log.warning('Cannot detect frame size from the dataset, use default window size: (1333, 800).')
        player = Player(win_size=win_size, win_name=str(seq), model=tracker, output_dir=output_dir, _log=_log.info)

        # frame range
        start_frame = int(frame_range['start'] * len(seq))
        end_frame = int(frame_range['end'] * len(seq))

        # load data sequence
        seq_loader = DataLoader(torch.utils.data.Subset(seq, range(start_frame, end_frame)))
        num_frames += len(seq_loader)

        # load previous recorded results (replay)
        results = seq.load_results(load_results_dir)

        if not results:
            start = time.time()
            # iterating data sequence
            for frame_id, frame_data in enumerate(tqdm.tqdm(seq_loader, file=sys.stdout)):
                with torch.no_grad():
                    # get original image for plotting
                    img = frame_data['orig_img'].detach().numpy()[0]
                    # model forwarding
                    if not player.step(frame_data, img):
                        _log.info(f"Skipped SEQ: {seq}.")
                        break

            time_total += time.time() - start

            # print info
            results = tracker.get_results()
            _log.info(f"NUM TRACKS: {len(results)} ReIDs: {tracker.num_reids}")
            _log.info(f"RUNTIME: {time.time() - start :.2f} s")
        else:
            # replay
            _log.info("LOAD RESULTS")
            player.delay = 42
            for frame_id, frame_data in enumerate(tqdm.tqdm(seq_loader, file=sys.stdout)):
                img = frame_data['orig_img'].detach().numpy()[0]
                if not player.interpret_results(frame_id, img, results):
                    _log.info(f"Skipped SEQ: {seq}.")
                    break

        # reset player
        del player

    if time_total:
        _log.info(f"RUNTIME ALL SEQS (w/o EVAL or IMG WRITE): "
                  f"{time_total:.2f} s for {num_frames} frames "
                  f"({num_frames / time_total:.2f} Hz)")
    return
