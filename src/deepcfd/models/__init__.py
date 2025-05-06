from .UNet import UNet
from .UNetEx import UNetEx
from .UNetExAvg import UNetExAvg
from .UNetExMod import UNetExMod


def get_model(key):
    if key.lower() == 'unet':
        return UNet
    elif key.lower() == 'unetex':
        return UNetEx
    elif key.lower() == 'unetexavg':
        return UNetExAvg
    elif key.lower() == 'unetexmod':
        return UNetExMod
    else:
        raise NotImplementedError
    
    