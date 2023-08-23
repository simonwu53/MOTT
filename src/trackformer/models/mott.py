# Copyright 2023 Shan Wu
# Modification by Shan
# * Added the proposed MOTT model
# * Simplify the model architecture in a single model class
# * Utilize customized CSWIN and Deformabnle Decoder module
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import copy
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from contextlib import nullcontext
from typing import Union, List
from ..util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list, get_clones
from .matcher import HungarianMatcher


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MOTT(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, encoder, decoder, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, overflow_boxes=False,
                 merge_frame_features=False,
                 track_query_false_positive_prob: float = 0.0,
                 track_query_false_negative_prob: float = 0.0,
                 matcher: HungarianMatcher = None,
                 backprop_prev_frame=False):
        """ Initializes the model.
        Parameters:
            encoder: torch module of the backbone to be used. See backbone.py
            decoder: torch module of the transformer architecture. See deformable.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal
                         number of objects DETR can detect in a single image. For COCO,
                         we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        # variables for training
        self._matcher = matcher
        self._track_query_false_positive_prob = track_query_false_positive_prob
        self._track_query_false_negative_prob = track_query_false_negative_prob
        self._backprop_prev_frame = backprop_prev_frame

        # model params
        self._tracking = False
        self.num_queries = num_queries
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_dim = self.decoder.d_model
        self.overflow_boxes = overflow_boxes
        self.class_embed = nn.Linear(self.hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.aux_loss = aux_loss

        self.merge_frame_features = merge_frame_features
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, self.hidden_dim * 2)

        self.fpn_channels = encoder.num_channels[:3][::-1]  # TODO: Used in Segmentation, *REMOVE LATER*
        num_channels = encoder.num_channels[-3:]
        if num_feature_levels > 1:
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            num_backbone_outs = len(encoder.strides) - 1

            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.hidden_dim),
                ))
                in_channels = self.hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(num_channels[0], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )])
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones_like(self.class_embed.bias) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for
        # region proposal generation
        num_pred = self.decoder.num_layers
        if two_stage:
            num_pred += 1

        if with_box_refine:
            self.class_embed = get_clones(self.class_embed, num_pred)
            self.bbox_embed = get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        if self.merge_frame_features:
            merge_layer = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim, kernel_size=1)
            self.merge_features = get_clones(merge_layer, num_feature_levels)
        return

    def train(self, mode: bool = True):
        """Sets the module in train mode."""
        self._tracking = False
        return super().train(mode)

    def tracking(self):
        """Sets the module in tracking mode."""
        self.eval()
        self._tracking = True
        return self

    @staticmethod
    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def add_track_queries_to_targets(self, targets, prev_indices, prev_out, add_false_pos=True):
        device = prev_out['pred_boxes'].device

        # for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
        min_prev_target_ind = min([len(prev_ind[1]) for prev_ind in prev_indices])
        num_prev_target_ind = 0
        if min_prev_target_ind:
            num_prev_target_ind = torch.randint(0, min_prev_target_ind + 1, (1,)).item()

        num_prev_target_ind_for_fps = 0
        if num_prev_target_ind:
            num_prev_target_ind_for_fps = \
                torch.randint(int(math.ceil(self._track_query_false_positive_prob * num_prev_target_ind)) + 1,
                              (1,)).item()

        for i, (target, prev_ind) in enumerate(zip(targets, prev_indices)):
            prev_out_ind, prev_target_ind = prev_ind

            # random subset
            if self._track_query_false_negative_prob:  # and len(prev_target_ind):
                # random_subset_mask = torch.empty(len(prev_target_ind)).uniform_()
                # random_subset_mask = random_subset_mask.ge(
                #     self._track_query_false_negative_prob)

                # random_subset_mask = torch.randperm(len(prev_target_ind))[:torch.randint(0, len(prev_target_ind) + 1, (1,))]
                random_subset_mask = torch.randperm(len(prev_target_ind))[:num_prev_target_ind]

                # if not len(random_subset_mask):
                #     target['track_query_hs_embeds'] = torch.zeros(0, self.hidden_dim).float().to(device)
                #     target['track_queries_placeholder_mask'] = torch.zeros(self.num_queries).bool().to(device)
                #     target['track_queries_mask'] = torch.zeros(self.num_queries).bool().to(device)
                #     target['track_queries_fal_pos_mask'] = torch.zeros(self.num_queries).bool().to(device)
                #     target['track_query_boxes'] = torch.zeros(0, 4).to(device)
                #     target['track_query_match_ids'] = torch.tensor([]).long().to(device)

                #     continue

                prev_out_ind = prev_out_ind[random_subset_mask]
                prev_target_ind = prev_target_ind[random_subset_mask]

            # detected prev frame tracks
            prev_track_ids = target['prev_target']['track_ids'][prev_target_ind]

            # match track ids between frames
            target_ind_match_matrix = prev_track_ids.unsqueeze(dim=1).eq(target['track_ids'])
            target_ind_matching = target_ind_match_matrix.any(dim=1)
            target_ind_matched_idx = target_ind_match_matrix.nonzero()[:, 1]

            # current frame track ids detected in the prev frame
            # track_ids = target['track_ids'][target_ind_matched_idx]

            # index of prev frame detection in current frame box list
            target['track_query_match_ids'] = target_ind_matched_idx

            # random false positives
            if add_false_pos:
                prev_boxes_matched = prev_out['pred_boxes'][i, prev_out_ind[target_ind_matching]]

                not_prev_out_ind = torch.arange(prev_out['pred_boxes'].shape[1])
                not_prev_out_ind = [
                    ind.item()
                    for ind in not_prev_out_ind
                    if ind not in prev_out_ind]

                random_false_out_ind = []

                prev_target_ind_for_fps = torch.randperm(num_prev_target_ind)[:num_prev_target_ind_for_fps]

                # for j, prev_box_matched in enumerate(prev_boxes_matched):
                #     if j not in prev_target_ind_for_fps:
                #         continue

                for j in prev_target_ind_for_fps:
                    # if random.uniform(0, 1) < self._track_query_false_positive_prob:
                    prev_boxes_unmatched = prev_out['pred_boxes'][i, not_prev_out_ind]

                    # only cxcy
                    # box_dists = prev_box_matched[:2].sub(prev_boxes_unmatched[:, :2]).abs()
                    # box_dists = box_dists.pow(2).sum(dim=-1).sqrt()
                    # box_weights = 1.0 / box_dists.add(1e-8)

                    # prev_box_ious, _ = box_ops.box_iou(
                    #     box_ops.box_cxcywh_to_xyxy(prev_box_matched.unsqueeze(dim=0)),
                    #     box_ops.box_cxcywh_to_xyxy(prev_boxes_unmatched))
                    # box_weights = prev_box_ious[0]

                    # dist = sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

                    if len(prev_boxes_matched) > j:
                        prev_box_matched = prev_boxes_matched[j]
                        box_weights = \
                            prev_box_matched.unsqueeze(dim=0)[:, :2] - \
                            prev_boxes_unmatched[:, :2]
                        box_weights = box_weights[:, 0] ** 2 + box_weights[:, 0] ** 2
                        box_weights = torch.sqrt(box_weights)

                        # if box_weights.gt(0.0).any():
                        # if box_weights.gt(0.0).any():
                        random_false_out_idx = not_prev_out_ind.pop(
                            torch.multinomial(box_weights.cpu(), 1).item())
                    else:
                        random_false_out_idx = not_prev_out_ind.pop(torch.randperm(len(not_prev_out_ind))[0])

                    random_false_out_ind.append(random_false_out_idx)

                prev_out_ind = torch.tensor(prev_out_ind.tolist() + random_false_out_ind).long()

                target_ind_matching = torch.cat([
                    target_ind_matching,
                    torch.tensor([False, ] * len(random_false_out_ind)).bool().to(device)
                ])

            # MSDeformAttn can not deal with empty inputs therefore we
            # add single false pos to have at least one track query per sample
            # not_prev_out_ind = torch.tensor([
            #     ind
            #     for ind in torch.arange(prev_out['pred_boxes'].shape[1])
            #     if ind not in prev_out_ind])
            # false_samples_inds = torch.randperm(not_prev_out_ind.size(0))[:1]
            # false_samples = not_prev_out_ind[false_samples_inds]
            # prev_out_ind = torch.cat([prev_out_ind, false_samples])
            # target_ind_matching = torch.tensor(
            #     target_ind_matching.tolist() + [False, ]).bool().to(target_ind_matching.device)

            # track query masks
            track_queries_mask = torch.ones_like(target_ind_matching).bool()
            track_queries_fal_pos_mask = torch.zeros_like(target_ind_matching).bool()
            track_queries_fal_pos_mask[~target_ind_matching] = True

            # track_queries_match_mask = torch.ones_like(target_ind_matching).float()
            # matches indices with 1.0 and not matched -1.0
            # track_queries_mask[~target_ind_matching] = -1.0

            # set prev frame info
            target['track_query_hs_embeds'] = prev_out['hs_embed'][i, prev_out_ind]
            target['track_query_boxes'] = prev_out['pred_boxes'][i, prev_out_ind].detach()

            target['track_queries_mask'] = torch.cat([
                track_queries_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()

            target['track_queries_fal_pos_mask'] = torch.cat([
                track_queries_fal_pos_mask,
                torch.tensor([False, ] * self.num_queries).to(device)
            ]).bool()

        # add placeholder track queries to allow for batch sizes > 1
        # max_track_query_hs_embeds = max([len(t['track_query_hs_embeds']) for t in targets])
        # for i, target in enumerate(targets):

        #     num_add = max_track_query_hs_embeds - len(target['track_query_hs_embeds'])

        #     if not num_add:
        #         target['track_queries_placeholder_mask'] = torch.zeros_like(target['track_queries_mask']).bool()
        #         continue

        #     raise NotImplementedError

        #     target['track_query_hs_embeds'] = torch.cat(
        #         [torch.zeros(num_add, self.hidden_dim).to(device),
        #          target['track_query_hs_embeds']
        #     ])
        #     target['track_query_boxes'] = torch.cat(
        #         [torch.zeros(num_add, 4).to(device),
        #          target['track_query_boxes']
        #     ])

        #     target['track_queries_mask'] = torch.cat([
        #         torch.tensor([True, ] * num_add).to(device),
        #         target['track_queries_mask']
        #     ]).bool()

        #     target['track_queries_fal_pos_mask'] = torch.cat([
        #         torch.tensor([False, ] * num_add).to(device),
        #         target['track_queries_fal_pos_mask']
        #     ]).bool()

        #     target['track_queries_placeholder_mask'] = torch.zeros_like(target['track_queries_mask']).bool()
        #     target['track_queries_placeholder_mask'][:num_add] = True

    def forward_once(self, samples: Union[NestedTensor, List[Tensor]], targets: list = None, prev_features=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.encoder(samples)

        features_all = features
        features = features[-3:]
        pos = pos[-3:]

        if prev_features is None:
            prev_features = features
        else:
            prev_features = prev_features[-3:]

        memory_2d = []
        memory_1d = []
        masks = []
        spatial_shapes = []
        valid_ratios = []

        for l, (feat, prev_feat) in enumerate(zip(features, prev_features)):
            src, mask = feat.decompose()
            prev_src, _ = prev_feat.decompose()

            if self.merge_frame_features:
                src = self.merge_features[l](torch.cat([self.input_proj[l](src), self.input_proj[l](prev_src)], dim=1))
            else:
                src = self.input_proj[l](src)

            spatial_shapes.append(src.shape[-2:])
            memory_2d.append(src)
            memory_1d.append(src.flatten(2).transpose(1, 2))
            valid_ratios.append(self.get_valid_ratio(mask))
            masks.append(mask.flatten(1))

        if self.num_feature_levels > len(features):
            _len_srcs = len(features)

            for l in range(_len_srcs, self.num_feature_levels):  ### additional feat level
                if l == _len_srcs:
                    if self.merge_frame_features:
                        src = self.merge_features[l](torch.cat(
                            [self.input_proj[l](features[-1].tensors), self.input_proj[l](prev_features[-1].tensors)],
                            dim=1))
                    else:
                        src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](memory_2d[-1])

                m = features[-1].mask
                spatial_shape = src.shape[-2:]
                mask = F.interpolate(m[None].float(), size=spatial_shape).to(torch.bool)[0]
                pos_l = self.encoder[1](NestedTensor(src, mask)).to(src.dtype)

                memory_2d.append(src)
                memory_1d.append(src.flatten(2).transpose(1, 2))
                valid_ratios.append(self.get_valid_ratio(mask))
                masks.append(mask.flatten(1))
                spatial_shapes.append(spatial_shape)
                pos.append(pos_l)

        memory_flatten = torch.cat(memory_1d, 1)
        mask_flatten = torch.cat(masks, 1)
        valid_ratios = torch.stack(valid_ratios, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=memory_flatten.device)

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, memory, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.decoder(memory_flatten, mask_flatten, spatial_shapes, valid_ratios, query_embeds, targets)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               'hs_embed': hs[-1]}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        return out, targets, features_all, memory_2d, hs

    def forward(self, samples: NestedTensor, targets: list = None, prev_features=None):
        if targets is not None and not self._tracking:
            prev_targets = [target['prev_target'] for target in targets]

            # if self.training and random.uniform(0, 1) < 0.5:
            if self.training:
                # if True:
                backprop_context = torch.no_grad
                if self._backprop_prev_frame:
                    backprop_context = nullcontext

                with backprop_context():
                    if 'prev_prev_image' in targets[0]:
                        for target, prev_target in zip(targets, prev_targets):
                            prev_target['prev_target'] = target['prev_prev_target']

                        prev_prev_targets = [target['prev_prev_target'] for target in targets]

                        # PREV PREV
                        prev_prev_out, _, prev_prev_features, _, _ = self.forward_once(
                            [t['prev_prev_image'] for t in targets])

                        prev_prev_outputs_without_aux = {
                            k: v for k, v in prev_prev_out.items() if 'aux_outputs' not in k}
                        prev_prev_indices = self._matcher(prev_prev_outputs_without_aux, prev_prev_targets)

                        self.add_track_queries_to_targets(
                            prev_targets, prev_prev_indices, prev_prev_out, add_false_pos=False)

                        # PREV
                        prev_out, _, prev_features, _, _ = self.forward_once(
                            [t['prev_image'] for t in targets],
                            prev_targets,
                            prev_prev_features)
                    else:
                        prev_out, _, prev_features, _, _ = self.forward_once([t['prev_image'] for t in targets])

                    # prev_out = {k: v.detach() for k, v in prev_out.items() if torch.is_tensor(v)}

                    prev_outputs_without_aux = {
                        k: v for k, v in prev_out.items() if 'aux_outputs' not in k}
                    prev_indices = self._matcher(prev_outputs_without_aux, prev_targets)

                    self.add_track_queries_to_targets(targets, prev_indices, prev_out)
            else:
                # if not training we do not add track queries and evaluate detection performance only.
                # tracking performance is evaluated by the actual tracking evaluation.
                for target in targets:
                    device = target['boxes'].device

                    target['track_query_hs_embeds'] = torch.zeros(0, self.hidden_dim).float().to(device)
                    # target['track_queries_placeholder_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_queries_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_queries_fal_pos_mask'] = torch.zeros(self.num_queries).bool().to(device)
                    target['track_query_boxes'] = torch.zeros(0, 4).to(device)
                    target['track_query_match_ids'] = torch.tensor([]).long().to(device)

        out, targets, features, memory, hs = self.forward_once(samples, targets, prev_features)

        return out, targets, features, memory, hs

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
