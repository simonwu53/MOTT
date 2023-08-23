# Copyright 2023 Shan Wu
# Modification by Shan
# * Added a solely deformabnle decoder module instead of full model
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from torch import nn
from torch.nn.init import constant_, normal_, xavier_uniform_

from ..util.misc import inverse_sigmoid, get_clones, get_activation_fn
from ..util.box_ops import box_cxcywh_to_xyxy
from .ops.modules import MSDeformAttn


class DeformableDecoder(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_decoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_feature_levels = num_feature_levels
        self.num_layers = num_decoder_layers
        self.return_intermediate = return_intermediate_dec
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

        # decoder
        layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                  dropout, activation,
                                                  num_feature_levels, nhead, dec_n_points)
        self.layers = get_clones(layer, num_decoder_layers)

        # self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()
        return

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        # normal_(self.level_embed)
        return

    def forward(self, memory, mask_flatten, spatial_shapes, valid_ratios, query_embed=None, targets=None):
        assert self.two_stage or query_embed is not None

        # prepare input for decoder
        bs, _, c = memory.shape
        query_attn_mask = None

        # if self.two_stage:
        #     output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        #
        #     # hack implementation for two-stage Deformable DETR
        #     enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
        #     enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        #
        #     topk = self.two_stage_num_proposals
        #     topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        #     topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        #     topk_coords_unact = topk_coords_unact.detach()
        #     reference_points = topk_coords_unact.sigmoid()
        #     init_reference_out = reference_points
        #     pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        #     query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        # else:
        query_embed, tgt = torch.split(query_embed, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

        reference_points = self.reference_points(query_embed).sigmoid()

        if targets is not None and 'track_query_hs_embeds' in targets[0]:
            # print([t['track_query_hs_embeds'].shape for t in targets])
            # prev_hs_embed = torch.nn.utils.rnn.pad_sequence([t['track_query_hs_embeds'] for t in targets], batch_first=True, padding_value=float('nan'))
            # prev_boxes = torch.nn.utils.rnn.pad_sequence([t['track_query_boxes'] for t in targets], batch_first=True, padding_value=float('nan'))
            # print(prev_hs_embed.shape)
            # query_mask = torch.isnan(prev_hs_embed)
            # print(query_mask)

            prev_hs_embed = torch.stack([t['track_query_hs_embeds'] for t in targets])
            prev_boxes = torch.stack([t['track_query_boxes'] for t in targets])

            prev_query_embed = torch.zeros_like(prev_hs_embed)
            # prev_query_embed = self.track_query_embed.weight.expand_as(prev_hs_embed)
            # prev_query_embed = self.hs_embed_to_query_embed(prev_hs_embed)
            # prev_query_embed = None

            prev_tgt = prev_hs_embed
            # prev_tgt = self.hs_embed_to_tgt(prev_hs_embed)

            query_embed = torch.cat([prev_query_embed, query_embed], dim=1)
            tgt = torch.cat([prev_tgt, tgt], dim=1)

            reference_points = torch.cat([prev_boxes[..., :2], reference_points], dim=1)

            # if 'track_queries_placeholder_mask' in targets[0]:
            #     query_attn_mask = torch.stack([t['track_queries_placeholder_mask'] for t in targets])

        init_reference = reference_points

        # decoder
        # TODO: Disable query_embed
        # query_embed = None
        # hs, inter_references = self.model(
        #     tgt, reference_points, memory, spatial_shapes,
        #     valid_ratios, query_embed, mask_flatten, query_attn_mask)
        output = tgt
        intermediate = []
        intermediate_reference_points = []

        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]
            output = layer(output, query_embed, reference_points_input, memory, spatial_shapes, mask_flatten,
                           query_attn_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            hs, inter_references = torch.stack(intermediate), torch.stack(intermediate_reference_points)
        else:
            hs, inter_references = output, reference_points

        # if self.two_stage:
        #     return (hs, init_reference, inter_references,
        #             enc_outputf_class, enc_outputs_coord_unact)
        return hs, memory, init_reference, inter_references, None, None


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, src_padding_mask=None,
                query_attn_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = \
        self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), key_padding_mask=query_attn_mask)[
            0].transpose(0, 1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, src_padding_mask, query_attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformablePostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        ###
        # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        # scores = topk_values

        # topk_boxes = topk_indexes // out_logits.shape[2]
        # labels = topk_indexes % out_logits.shape[2]

        # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        ###

        scores, labels = prob.max(-1)
        # scores, labels = prob[..., 0:1].max(-1)
        boxes = box_cxcywh_to_xyxy(out_bbox)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [
            {'scores': s, 'scores_no_object': 1 - s, 'labels': l, 'boxes': b}
            for s, l, b in zip(scores, labels, boxes)]

        if results_mask is not None:
            for i, mask in enumerate(results_mask):
                for k, v in results[i].items():
                    results[i][k] = v[mask]

        return results


def build_deforamble_decoder(args):
    if 'motr' in args and args.motr:
        return DeformableDecoder(
            d_model=args.hidden_dim,
            nhead=args.nheads,
            num_decoder_layers=args.dec_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=args.num_feature_levels,
            dec_n_points=args.dec_n_points,
            two_stage=args.two_stage,
            two_stage_num_proposals=args.num_queries)
