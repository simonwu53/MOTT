from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .model import CSWinTransformer

import torch


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'cswin_224': _cfg(),
    'cswin_384': _cfg(
        crop_pct=1.0
    )
}


### 224 models
def CSWin_64_12211_tiny_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[1, 2, 21, 1],
                             split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model


def CSWin_64_24322_small_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=64, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model


def CSWin_96_24322_base_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 7, 7], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model


def CSWin_144_24322_large_224(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 7, 7], num_heads=[6, 12, 24, 24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_224']
    return model


### 384 models
def CSWin_96_24322_base_384(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 12, 12], num_heads=[4, 8, 16, 32], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model


def CSWin_144_24322_large_384(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=144, depth=[2, 4, 32, 2],
                             split_size=[1, 2, 12, 12], num_heads=[6, 12, 24, 24], mlp_ratio=4., **kwargs)
    model.default_cfg = default_cfgs['cswin_384']
    return model


"""
TESTING
"""
def build_tiny_model():
    from collections import OrderedDict

    # build model
    model = CSWin_64_12211_tiny_224()
    print(f'[CSWin] Num trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6} M')

    # load checkpoint
    ckpt = torch.load('ckpt/cswin_tiny_224.pth', map_location='cpu')
    if 'state_dict_ema' in ckpt:
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict_ema'].items():
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    return model


def test_model(model):
    # feed forward
    model.cuda()
    print('testing shape image size 224, shape (5, 3, 224, 224)')
    x = torch.rand(5, 3, 224, 224)  # (5, 3, 224, 224)
    y = model(x.cuda())
    print(f'output shape: {y.shape}')  # (5, 1000)

    del x, y

    # TODO: img size conflict with split size!!! (e.g. 512//32 != int, error in img2windows)
    print('testing shape image size 224, shape (5, 3, 448, 448)')
    x = torch.rand(5, 3, 448, 448)  # (5, 3, 224, 224)
    y = model(x.cuda())
    print(f'output shape: {y.shape}')  # (5, 1000)
    return


def test_img(model):
    from PIL import Image
    import torch.nn.functional as F
    import numpy as np

    im = Image.open('dataset/cnn_embed_full_1k.jpg')
    imnp = np.array(im)
    x = torch.tensor(imnp[:50, 7 * 50:8 * 50])  # [50, 50, 3]
    out = model(F.interpolate(x.unsqueeze(0).float().permute(0, 3, 1, 2).cuda(), size=224, mode='bilinear'))
    return
