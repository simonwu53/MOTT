# Copyright 2023 Shan Wu
# Mofification by Shan
# * Added bakcbone mode for other downstream projects
# * Added ability to accept arbitrary input sizes for backbone usage
# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------


import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np
from collections import OrderedDict

from .ops import DropPath, Merge_Block, pad_inputs, remove_padding, calc_conv_out_shape, calc_sw_pad
from .mlp import Mlp
from .attn import LePEAttention


class CSWinBlock(nn.Module):
    def __init__(self, dim, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x, spatial_shape):
        """
        x: B, H*W, C
        """

        # H = W = self.patches_resolution
        H, W = spatial_shape
        B, L, C = x.shape
        assert L == H * W, f"flatten img_tokens has wrong size. L={L}, H={H}, W={W}, HW={H * W}"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2], spatial_shape)
            x2 = self.attns[1](qkv[:, :, :, C // 2:], spatial_shape)
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv, spatial_shape)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, patch_size=16, in_channels=3, num_classes=1000, embed_dim=96, depth=[2, 2, 6, 2],
                 split_size=[3, 5, 7], num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, use_chk=False,
                 return_interm_layers=False, backbone_mode=False):
        super().__init__()
        self.use_chk = use_chk
        if use_chk:
            print('checkpointing used for CSWin.')

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.split_size = split_size
        heads = num_heads

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])

        self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
                for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)

        self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[-1])])

        self.norm = norm_layer(curr_dim)
        # Classifier head
        self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()

        # intermediate layers
        self.return_interm_layers = return_interm_layers
        self.backbone_mode = backbone_mode
        if backbone_mode:
            if return_interm_layers:
                # return_layers = { #stage : "out_name" }
                self.return_layers = {1: '0', 2: '1', 3: '2', 4: '3'}  # output specified layers
                self.strides = [4, 8, 16, 32]
                self.num_channels = [64, 128, 256, 512]
            else:
                self.return_layers = {4: '0'}  # output last feature
                self.strides = [32]
                self.num_channels = [512]

        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
            print('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head = self.head.cuda()
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def is_input_valid(self, h, w):
        # input shape check
        for i in range(4):  # 4 stages
            _h, _w = h // (2 ** (2 + i)), w // (2 ** (2 + i))
            if not (_h / self.split_size[i]).is_integer():
                print(f'input height dimension cannot be divided by split size at level {i + 1}')
                return 0
            if not (_w / self.split_size[i]).is_integer():
                print(f'input width dimension cannot be divided by split size at level {i + 1}')
                return 0
        return 1

    def forward_features(self, x):
        """
        original implementation of feature forwarding
        """
        B, C, H, W = x.shape

        # input shape check
        if not self.is_input_valid(H, W):
            print('input tensor invalid. Can not proceed.')
            return

        # input layers
        x = self.stage1_conv_embed(x)
        H, W = H // 4, W // 4

        # stage 1
        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, (H, W))
            else:
                x = blk(x, (H, W))

        # remaining stages: 2, 3, 4
        for i, (pre, blocks) in enumerate(zip([self.merge1, self.merge2, self.merge3],
                                              [self.stage2, self.stage3, self.stage4])):
            x = pre(x, (H, W))
            H, W = H // 2, W // 2
            for blk in blocks:
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x, (H, W))
                else:
                    x = blk(x, (H, W))

        x = self.norm(x)
        return torch.mean(x, dim=1)

    def forward_backbone(self, x):
        """
        optimized feature forwarding with arbitrary input sizes using a padding hack
        * optimized/clearer workflow for four stages
        * use accurate calculation of down-sampling size
        * adds a padding hack for arbitrary img sizes
        """
        B, C, H, W = x.shape
        out = OrderedDict()

        # input layers
        x = self.stage1_conv_embed(x)
        H, W = calc_conv_out_shape((H, W), 7, 4, 2)

        # four stages
        for i in range(4):
            stage = i + 1

            # padding hack
            pad = [0, 0, 0, 0]  # padding on original shape
            pad_rev = [0, W, 0, H]  # reverse padding for restoring original shape
            H_adj, W_adj = H, W  # new shape in case of padding
            if not (W / self.split_size[i]).is_integer() and stage < 4:
                left_pad, right_pad = calc_sw_pad(W, self.split_size[i])
                pad[:2] = (left_pad, right_pad)
                W_adj = W + left_pad + right_pad
                pad_rev[:2] = (left_pad, W_adj - right_pad)
            if not (H / self.split_size[i]).is_integer() and stage < 4:
                top_pad, bottom_pad = calc_sw_pad(H, self.split_size[i])
                pad[2:] = (top_pad, bottom_pad)
                H_adj = H + top_pad + bottom_pad
                pad_rev[2:] = (top_pad, H_adj - bottom_pad)
            if sum(pad) != 0:
                x = pad_inputs(x, (H, W), pad)

            # forward stage
            for blk in getattr(self, f'stage{stage}'):
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x, (H_adj, W_adj))
                else:
                    x = blk(x, (H_adj, W_adj))

            # recover to original HW
            if sum(pad) != 0:
                # TODO: whether to keep this mechanism?
                x = remove_padding(x, (H_adj, W_adj), pad_rev)

            # collect intermediate output
            if stage in self.return_layers:
                out[self.return_layers[stage]] = x.transpose(-2, -1).contiguous().view(B, -1, H, W)

            # downsampling
            if stage < 4:
                x = getattr(self, f'merge{stage}')(x, (H, W))
                H, W = calc_conv_out_shape((H, W), 3, 2, 1)

        # output collected features
        # TODO: LayerNorm for the last stage output?
        return out

    def forward(self, x):
        if self.backbone_mode:
            return self.forward_backbone(x)
        else:
            x = self.forward_features(x)
            x = self.head(x)
        return x
