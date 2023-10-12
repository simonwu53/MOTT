# MOTT: A New Model for Multi-Object Tracking Based on Green Learning Paradigm

This is the official implementation of MOTT paper, a novel multi-object tracking model. 
The code is inspired
by [TrackFormer](https://github.com/timmeinhardt/trackformer), [TransTrack](https://github.com/PeizeSun/TransTrack),
[DETR](https://github.com/fundamentalvision/Deformable-DETR), [CSWin](https://github.com/microsoft/CSWin-Transformer)
by taking the effective Transformer components (CSWin Encoder, deformable DETR decoder) forming a new light-weighted
Transformer specialized in MOT.

**Considering that the manuscript is currently under review for publication, this repository will be subject to ongoing updates.**

<div align="center">
    <img src="assets/MOT17-03-mott.gif" alt="MOT17-03" width="375"/>
    <img src="assets/DT-054-mott.gif" alt="DanceTrack-054" width="375"/>
    <img src="assets/MOT20-08-mott.gif" alt="MOTS20-08" width="375"/>
</div>

## Motivation

Multi-object tracking (MOT) is one of the most essential and challenging tasks in computer vision (CV). Unlike object
detectors, MOT systems nowadays are more complicated and consist of several neural network models. Thus, the balance
between the system performance and the runtime is crucial for online scenarios. While some of the works contribute by
adding more modules to achieve improvements, we propose a pruned model by leveraging the state-of-the-art Transformer
backbone model. Our model saves up to 62% FLOPS compared with other Transformer-based models and almost as twice as
fast as them. The results of the proposed model are still competitive among the state-of-the-art methods. Moreover, we
will open-source our modified Transformer backbone model for general CV tasks as well as the MOT system.

<div align="center">
   <img src="assets/model_arch.png" alt="MOTT-model" width="560"/>
</div>

## Installation

Please visit the [installation.md](docs/installation.md) for guidances.

## Training

Please visit the [dataset.md](docs/dataset.md) for dataset preparation. Then, head to the [train.md](docs/train.md) for
training scripts.

## Evaluation

### MOT Evaluation

We split the MOT17 dataset into two halves as shown in the paper, then we trained all models on the first half using the same schedule and evaluated on the second half.

<center>

|    Model    | MOTA  | MOTP  | IDF1  | MT  | ML |
|:-----------:|:-----:|:-----:|:-----:|:---:|:--:|
| TransTrack  | 66.5% | 83.4% | 66.8% | 134 | 61 |
| TrackFormer | 67.0% | 84.1% | 69.5% | 152 | 57 |
|  **MOTT**   | 71.6% | 84.5% | 71.7% | 166 | 41 |

</center>

```bash
# TODO
```

### Test your own videos

1. Install and activate the Python environment.
2. Download the pre-trained weights `cswin_tiny_224.pth` and `mot17_ch_mott.tar.gz`
   from [OwnCloud](https://owncloud.ut.ee/owncloud/s/wppiGAgSHTxEdJ8).
3. Put `cswin_tiny_224.pth` in `./models` folder. Extract `mot17_ch_mott` folder and put it in `./models`
   folder.
4. Put the testing video (`.mov`, `.mp4`, `.avi` formats) in `./data/videos/` folder.
5. Run the command at the root of the repo:

```bash
bash scripts/run_online.sh
```

The program will show a list of available videos in the folder.
Select a video by inputting the index number.
Stop video by issuing key `q`.
Terminate the program by issuing `ctrl+c`.

The config file of the program is stored in `cfgs/track_online.yaml`.

## Contributors
Shan Wu; Amnir Hadachi; Chaoru Lu, Damien Vivet. 

## Citation
If you use NetCalib in an academic work, please cite:
```
@article{wu2023mott,
  title={MOTT: A new model for multi-object tracking based on green learning paradigm},
  author={Wu, Shan and Hadachi, Amnir and Lu, Chaoru and Vivet, Damien},
  journal={AI Open},
  year={2023},
  publisher={Elsevier}
}
```

Published paper is [here](https://www.sciencedirect.com/science/article/pii/S2666651023000165).
