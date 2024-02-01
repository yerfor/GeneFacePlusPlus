import numpy as np
import cv2
from utils.commons.tensor_utils import convert_to_np


def plot_attention_img(attention_img, color_bar='jet'):
    """
    attention_img: raw attention in network, tensor or array, in 0~1 scale, shape [H, W,]
    color_bar: jet, summer, etc see this https://blog.csdn.net/loveliuzz/article/details/73648505
    return: ready-to-visualize attention img in -1~1 scale.
    """
    attention_img = convert_to_np(attention_img)
    assert attention_img.ndim == 2
    attention_img = np.uint8(255 * attention_img)
    color_bar_dict = {
        'jet': cv2.COLORMAP_JET,
        'summer': cv2.COLORMAP_SUMMER,
        'hot': cv2.COLORMAP_HOT
    }
    color_bar = color_bar_dict.get(color_bar, getattr(cv2, f"COLORMAP_{color_bar.upper()}"))
    attention_img = cv2.applyColorMap(attention_img, color_bar) / 127.5 - 1
    attention_img = attention_img[:, :, ::-1] # flip RGB
    return attention_img