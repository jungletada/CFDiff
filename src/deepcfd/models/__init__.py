import torch
from timm.models.layers import StdConv2dSame

from .UNet import UNet
from .UNetEx import UNetEx
from .UNetExAvg import UNetExAvg
from .UNetExMod import UNetExMod
from .dpt_models import DPTDepthModel


def build_model(key):
    if key.lower() == 'unet':
        return UNet(in_channels=2, out_channels=3)
    
    elif key.lower() == 'unetex':
        return UNetEx(in_channels=2, out_channels=3)
    
    elif key.lower() == 'unetexavg':
        return UNetExAvg(in_channels=2, out_channels=3)
    
    elif key.lower() == 'unetexmod':
        return UNetExMod(in_channels=2, out_channels=3)
    
    elif key.lower() == 'dpt':
        model = DPTDepthModel(
            path=None,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        print(f'Original stem conv: {model.pretrained.model.patch_embed.backbone.stem.conv}')
        with torch.no_grad():
            model.pretrained.model.patch_embed.backbone.stem.conv = StdConv2dSame(
                                        2, 64, kernel_size=(7, 7), stride=(2, 2), bias=False)
            model.pretrained.model.patch_embed.backbone.stem.conv.weight.data = \
                model.pretrained.model.patch_embed.backbone.stem.conv.weight.data.contiguous()
        print(f'Modified stem conv: {model.pretrained.model.patch_embed.backbone.stem.conv}')
        return model
    else:
        raise NotImplementedError
    
    