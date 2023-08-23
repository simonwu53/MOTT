# Copyright 2023 Shan Wu
# Modification by Shan
# * Added features for proposed MOTT model
# * Added weights migration function for MOTT model
# ---
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import re
import yaml
import shutil
import torch
from argparse import Namespace
from typing import Optional, Union

from .backbone import build_backbone
from .deformable import DeformablePostProcess, build_deforamble_decoder
from .criterion import SetCriterion
from .matcher import build_matcher
from .mott import MOTT
# from .segmentation import MOTTSegmTracking, PostProcessSegm, PostProcessPanoptic
from ..util.misc import nested_dict_to_namespace

MODEL_VERSION: int = 3


def build_model(args):
    # DISABLE MASK HEAD
    if args.masks:
        print(f"Currently Mask head is not available for MOTT.")
        exit(0)

    if args.dataset == 'coco':
        num_classes = 91
    elif args.dataset == 'coco_panoptic':
        num_classes = 250
    elif args.dataset in ['coco_person', 'mot', 'mot_crowdhuman', 'crowdhuman', 'mot_coco_person']:
        # num_classes = 91
        num_classes = 20
        # num_classes = 1
    else:
        raise NotImplementedError

    if 'motr' in args and args.motr:
        device = torch.device(args.device)
        matcher = build_matcher(args)
        backbone = build_backbone(args)
        decoder = build_deforamble_decoder(args)

        mask_kwargs = {
            'freeze_mott': getattr(args, 'freeze_mott', False),
            'parallel': getattr(args, 'parallel_mask_head', False),
        }

        mott_kwargs = {
            'encoder': backbone,
            'decoder': decoder,
            'num_feature_levels': args.num_feature_levels,
            'merge_frame_features': args.merge_frame_features,
            'num_classes': num_classes - 1 if args.focal_loss else num_classes,
            'num_queries': args.num_queries,
            'with_box_refine': args.with_box_refine,
            'aux_loss': args.aux_loss,
            'overflow_boxes': args.overflow_boxes,
            'two_stage': args.two_stage,
            'track_query_false_positive_prob': args.track_query_false_positive_prob,
            'track_query_false_negative_prob': args.track_query_false_negative_prob,
            'matcher': matcher,
            'backprop_prev_frame': args.track_backprop_prev_frame
        }

        if args.tracking:
            model = MOTT(**mott_kwargs)
        else:
            raise NotImplementedError('detection model not implemented.')

    # DISABLE OTHER UNNECESSARY MODELS
    else:
        print(f"Currently other models are not available.")
        exit(0)

    print(f'Model Class: {model.__class__.__name__}.')

    weight_dict = {'loss_ce': args.cls_loss_coef,
                   'loss_bbox': args.bbox_loss_coef,
                   'loss_giou': args.giou_loss_coef, }

    # if args.masks:
    #     weight_dict["loss_mask"] = args.mask_loss_coef
    #     weight_dict["loss_dice"] = args.dice_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    # if args.masks:
    #     losses.append('masks')

    criterion = SetCriterion(
        num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        focal_loss=args.focal_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tracking=args.tracking,
        track_query_false_positive_eos_weight=args.track_query_false_positive_eos_weight, )
    criterion.to(device)
    postprocessors = {'bbox': DeformablePostProcess()}
    # if args.masks:
    #     postprocessors['segm'] = PostProcessSegm()
    #     if args.dataset == "coco_panoptic":
    #         is_thing_map = {i: i <= 90 for i in range(201)}
    #         postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)
    #     pass

    return model, criterion, postprocessors


def initialize_model(model_path: str, _logger: Optional = None, _cuda: bool = True, _tracking: bool = True,
                     ckpt_version: int = MODEL_VERSION, **kwargs):
    # find config file path
    obj_detect_config_path = os.path.join(os.path.dirname(model_path), 'config.yaml')
    # load config dict
    obj_detect_args = nested_dict_to_namespace({**yaml.unsafe_load(open(obj_detect_config_path)), **kwargs})
    # load img transform
    img_transform = obj_detect_args.img_transform
    # build model
    mott, _, mott_post = build_model(obj_detect_args)
    # load model checkpoint
    obj_detect_checkpoint = torch.load(model_path, map_location='cpu')
    # get model state dict
    obj_detect_state_dict = obj_detect_checkpoint['model']
    # adapt to new architecture
    if ckpt_version != MODEL_VERSION:
        if ckpt_version == 1 and MODEL_VERSION == 3:
            new_state_dict = upgrade_model_weights_v1_3(obj_detect_state_dict)
        else:
            raise AttributeError(f'No mapping from Ver.{ckpt_version} weights to Ver.{MODEL_VERSION}')
        obj_detect_state_dict = new_state_dict
    # load state dict
    missing_keys, unexpected_keys = mott.load_state_dict(obj_detect_state_dict, strict=False)
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        print(f"State dict partially loaded. \n\tMissing Keys: {missing_keys} \n\tUnexpected Keys: {unexpected_keys}")
    if 'epoch' in obj_detect_checkpoint:
        if _logger is not None:
            _logger.info(f"INIT object detector [EPOCH: {obj_detect_checkpoint['epoch']}]")

    # post process
    n_parameters = sum(p.numel() for p in mott.parameters() if p.requires_grad)
    print('NUM TRAINABLE MODEL PARAMS:', n_parameters)
    if _cuda:
        mott.to('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Moved model to GPU.')
    if hasattr(mott, 'tracking') and _tracking:
        mott.tracking()
        print('Set tracking mode.')
    return mott, mott_post, img_transform


def upgrade_model_weights_v1_3(model: Union[str, dict]):
    """
    param model: either model's checkpoint's path in string OR model's state dict object
    """
    print('Adapt Ver.1 model weights to Ver.3')
    if isinstance(model, str):
        checkpoint = torch.load(model, map_location='cpu')
        state_dict = checkpoint['model']
    else:
        state_dict = model
    new_state_dict = {}
    pattern1 = re.compile(r'^transformer(?:\.decoder)?')
    pattern2 = re.compile(r'^backbone')
    for k, v in state_dict.items():
        if pattern1.match(k):
            new_state_dict[pattern1.sub('decoder', k)] = v
        elif pattern2.match(k):
            new_state_dict[pattern2.sub('encoder', k)] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def upgrade_ckpt_v1_3(old_model_path: str, new_model_path: str, copy_config: bool = True):
    # check files
    if not os.path.exists(old_model_path):
        raise FileNotFoundError(f'Old model path not existing? {old_model_path}')
    if not os.path.exists(os.path.dirname(new_model_path)):
        os.makedirs(os.path.dirname(new_model_path))

    # upgrade state dict
    checkpoint = torch.load(old_model_path, map_location='cpu')
    new_state_dict = upgrade_model_weights_v1_3(checkpoint['model'])
    checkpoint['model'] = new_state_dict
    torch.save(checkpoint, new_model_path)

    # copy config
    if copy_config:
        old_config_path = os.path.join(os.path.dirname(old_model_path), 'config.yaml')
        new_config_path = os.path.join(os.path.dirname(new_model_path), 'config.yaml')
        shutil.copy(old_config_path, new_config_path)
    return
