import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_sw_pad(l, sw):
    total_pad = sw - l % sw
    left = total_pad // 2
    right = total_pad - left
    return left, right


def calc_conv_out_shape(spatial_shape, kernel_size, stride, padding):
    H_in, W_in = spatial_shape
    H_out = (H_in-kernel_size+2*padding) // stride + 1
    W_out = (W_in-kernel_size+2*padding) // stride + 1
    return H_out, W_out


def pad_inputs(x, spatial_shape, pad):
    B, L, C = x.shape
    H, W = spatial_shape

    x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
    x = F.pad(x, pad, value=0)
    x = x.reshape(B, C, -1).transpose(-2, -1).contiguous()
    return x


def remove_padding(x, spatial_shape, pad_rev):
    B, L, C = x.shape
    H, W = spatial_shape
    left, right, top, bottom = pad_rev

    x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
    x = x[:, :, top:bottom, left:right]
    x = x.reshape(B, C, -1).transpose(-2, -1).contiguous()
    return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x, spatial_shape):
        H, W = spatial_shape
        B, new_HW, C = x.shape
        assert new_HW == H * W, f"spatial shape mismatch, L={new_HW}, H={H}, W={W}, HW={H*W}"
        # H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x
