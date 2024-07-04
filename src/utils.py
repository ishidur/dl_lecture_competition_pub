import random
from enum import Enum, auto

import numpy as np
import torch


class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()


class EventRepresentation:
    def __init__(self):
        pass

    def convert(self, events):
        raise NotImplementedError


class VoxelGrid(EventRepresentation):
    def __init__(self, input_size: tuple, normalize: bool):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros((input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]
        self.normalize = normalize

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events["p"].device)
            voxel_grid = self.voxel_grid.clone()

            t_norm = events["t"]
            t_norm = (C - 1) * (t_norm - t_norm[0]) / (t_norm[-1] - t_norm[0])

            x0 = events["x"].int()
            y0 = events["y"].int()
            t0 = t_norm.int()

            value = 2 * events["p"] - 1
            # start_t = time()
            for xlim in [x0, x0 + 1]:
                for ylim in [y0, y0 + 1]:
                    for tlim in [t0, t0 + 1]:
                        mask = (
                            (xlim < W)
                            & (xlim >= 0)
                            & (ylim < H)
                            & (ylim >= 0)
                            & (tlim >= 0)
                            & (tlim < self.nb_channels)
                        )
                        interp_weights = (
                            value
                            * (1 - (xlim - events["x"]).abs())
                            * (1 - (ylim - events["y"]).abs())
                            * (1 - (tlim - t_norm).abs())
                        )
                        index = H * W * tlim.long() + W * ylim.long() + xlim.long()

                        voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

            if self.normalize:
                mask = torch.nonzero(voxel_grid, as_tuple=True)
                if mask[0].size()[0] > 0:
                    mean = voxel_grid[mask].mean()
                    std = voxel_grid[mask].std()
                    if std > 0:
                        voxel_grid[mask] = (voxel_grid[mask] - mean) / std
                    else:
                        voxel_grid[mask] = voxel_grid[mask] - mean

        return voxel_grid


class PolarityCount(EventRepresentation):
    def __init__(self, input_size: tuple):
        assert len(input_size) == 3
        self.voxel_grid = torch.zeros((input_size), dtype=torch.float, requires_grad=False)
        self.nb_channels = input_size[0]

    def convert(self, events):
        C, H, W = self.voxel_grid.shape
        with torch.no_grad():
            self.voxel_grid = self.voxel_grid.to(events["p"].device)
            voxel_grid = self.voxel_grid.clone()

            x0 = events["x"].int()
            y0 = events["y"].int()

            # start_t = time()
            for xlim in [x0, x0 + 1]:
                for ylim in [y0, y0 + 1]:
                    mask = (xlim < W) & (xlim >= 0) & (ylim < H) & (ylim >= 0)
                    interp_weights = (1 - (xlim - events["x"]).abs()) * (1 - (ylim - events["y"]).abs())
                    index = H * W * events["p"].long() + W * ylim.long() + xlim.long()

                    voxel_grid.put_(index[mask], interp_weights[mask], accumulate=True)

        return voxel_grid


def flow_16bit_to_float(flow_16bit: np.ndarray):
    assert flow_16bit.dtype == np.uint16
    assert flow_16bit.ndim == 3
    h, w, c = flow_16bit.shape
    assert c == 3

    valid2D = flow_16bit[..., 2] == 1
    assert valid2D.shape == (h, w)
    assert np.all(flow_16bit[~valid2D, -1] == 0)
    valid_map = np.where(valid2D)

    # to actually compute something useful:
    flow_16bit = flow_16bit.astype("float")

    flow_map = np.zeros((h, w, 2))
    flow_map[valid_map[0], valid_map[1], 0] = (flow_16bit[valid_map[0], valid_map[1], 0] - 2**15) / 128
    flow_map[valid_map[0], valid_map[1], 1] = (flow_16bit[valid_map[0], valid_map[1], 1] - 2**15) / 128
    return flow_map, valid2D


######################################


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def move_tensor_to_cuda(dict_tensors, gpu):
    assert isinstance(dict_tensors, dict)
    for key, value in dict_tensors.items():
        if isinstance(value, torch.Tensor):
            dict_tensors[key] = value.to(gpu, non_blocking=True)
    return dict_tensors


def move_dict_to_cuda(dictionary_of_tensors, gpu):
    if isinstance(dictionary_of_tensors, dict):
        return {
            key: move_dict_to_cuda(value, gpu)
            for key, value in dictionary_of_tensors.items()
            if isinstance(value, torch.Tensor)
        }
    return dictionary_of_tensors.to(gpu, dtype=torch.float)


def move_list_to_cuda(list_of_dicts, gpu):
    for i in range(len(list_of_dicts)):
        list_of_dicts[i] = move_tensor_to_cuda(list_of_dicts[i], gpu)
    return list_of_dicts


def move_batch_to_cuda(batch, gpu):
    if isinstance(batch, dict):
        return move_tensor_to_cuda(batch, gpu)
    elif isinstance(batch, list):
        return move_list_to_cuda(batch, gpu)
    else:
        raise Exception("Batch is not a list or dict")


def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    """
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    """
    np.save(f"{file_name}.npy", flow.cpu().numpy())
