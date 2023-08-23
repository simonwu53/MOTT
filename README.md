# MOTT: A New Model for Multi-Object Tracking Based on Green Learning Paradigm

This is the official implementation of MOTT paper, a novel multi-object tracking model. The code is inspired
by [TrackFormer](https://github.com/timmeinhardt/trackformer), [TransTrack](https://github.com/PeizeSun/TransTrack),
[DETR](https://github.com/fundamentalvision/Deformable-DETR), [CSWin](https://github.com/microsoft/CSWin-Transformer)
by taking the effective Transformer components (CSWin Encoder, deformable DETR decoder) forming a new light-weighted
Transformer specialized in MOT.

> TODO: image visualization

## Motivation

Multi-object tracking (MOT) is one of the most essential and challenging tasks in computer vision (CV). Unlike object
detectors, MOT systems nowadays are more complicated and consist of several neural network models. Thus, the balance
between the system performance and the runtime is crucial for online scenarios. While some of the works contribute by
adding more modules to achieve improvements, we propose a pruned model by leveraging the state-of-the-art Transformer
backbone model. Our model saves up to 62% FLOPS compared with other Transformer-based models and almost as twice as
fast as them. The results of the proposed model are still competitive among the state-of-the-art methods. Moreover, we
will open-source our modified Transformer backbone model for general CV tasks as well as the MOT system.

> TODO: Architecture image

## Prerequisites

Please visit the [installation.md](docs/installation.md) for guidances.

## Training

Please visit the [dataset.md](docs/dataset.md) for dataset preparation. Then, head to the [train.md](docs/train.md) for
training scripts.

## Evaluation

> TODO