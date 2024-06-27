import numpy as np
import torch
import torch.nn as nn
import re
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as TF


def dictionary_of_numpy_arrays_to_tensors(sample):
    """Transforms dictionary of numpy arrays to dictionary of tensors."""
    if isinstance(sample, dict):
        return {
            key: dictionary_of_numpy_arrays_to_tensors(value)
            for key, value in sample.items()
        }
    if isinstance(sample, np.ndarray):
        if len(sample.shape) == 2:
            return torch.from_numpy(sample).float().unsqueeze(0)
        else:
            return torch.from_numpy(sample).float()
    return sample


def apply_randomcrop_to_sample(sample, crop_size):
    """
    Applies a random crop to a sample.
    :param sample: a sample
    :param crop_size: the size of the crop
    :return: the cropped sample
    """
    i, j, h, w = RandomCrop.get_params(sample["event_volume"], output_size=crop_size)
    keys_to_crop = ["event_volume", "flow_gt", "reverse_flow_gt"]

    for key, value in sample.items():
        if key in keys_to_crop:
            if isinstance(value, torch.Tensor):
                sample[key] = TF.crop(value, i, j, h, w)
            elif isinstance(value, list) or isinstance(value, tuple):
                sample[key] = [TF.crop(v, i, j, h, w) for v in value]
    return sample


def downsample_spatial(x, factor):
    """
    Downsample a given tensor spatially by a factor.
    :param x: PyTorch tensor of shape [batch, num_bins, height, width]
    :param factor: downsampling factor
    :return: PyTorch tensor of shape [batch, num_bins, height/factor, width/factor]
    """
    assert factor > 0, "Factor must be positive!"

    assert x.shape[-1] % factor == 0, "Width of x must be divisible by factor!"
    assert x.shape[-2] % factor == 0, "Height of x must be divisible by factor!"

    return nn.AvgPool2d(kernel_size=factor, stride=factor)(x)


def downsample_spatial_mask(x, factor):
    """
    Downsample a given mask (boolean) spatially by a factor.
    :param x: PyTorch tensor of shape [batch, num_bins, height, width]
    :param factor: downsampling factor
    :return: PyTorch tensor of shape [batch, num_bins, height/factor, width/factor]
    """
    assert factor > 0, "Factor must be positive!"

    assert x.shape[-1] % factor == 0, "Width of x must be divisible by factor!"
    assert x.shape[-2] % factor == 0, "Height of x must be divisible by factor!"

    return nn.AvgPool2d(kernel_size=factor, stride=factor)(x.float()) >= 0.5
