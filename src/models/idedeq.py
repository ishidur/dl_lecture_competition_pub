import torch
import torch.nn as nn
from torch.nn.functional import unfold, grid_sample, interpolate

from .extractor import LiteEncoder
from .update import LiteUpdateBlock

from math import sqrt


class IDEDEQIDO(nn.Module):
    def __init__(self, config):
        super(IDEDEQIDO, self).__init__()
        self.hidden_dim = getattr(config, "hidden_dim", 96)
        self.input_dim = 64
        self.downsample = getattr(config, "downsample", 8)
        self.fnet = LiteEncoder(
            output_dim=self.input_dim // 2,
            dropout=0,
            n_first_channels=2,
            stride=2 if self.downsample == 8 else 1,
        )
        self.update_net = LiteUpdateBlock(
            hidden_dim=self.hidden_dim,
            input_dim=self.input_dim,
            num_outputs=1,
            downsample=self.downsample,
        )
        self.deblur_iters = config.update_iters
        self.hook = None
        self.deblur_mode = getattr(config, "deblur_mode", "voxel")
        self.cnet = LiteEncoder(
            output_dim=self.hidden_dim // 2,
            dropout=0,
            n_first_channels=2,
            stride=2 if self.downsample == 8 else 1,
        )

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        _, D, H, W = mask.shape
        upsample_ratio = int(sqrt(D / 9))
        mask = mask.view(N, 1, 9, upsample_ratio, upsample_ratio, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, upsample_ratio * H, upsample_ratio * W)

    @staticmethod
    def upflow8(flow, mode="bilinear"):
        new_size = (8 * flow.shape[-2], 8 * flow.shape[-1])
        return 8 * interpolate(flow, size=new_size, mode=mode, align_corners=True)

    @staticmethod
    def create_identity_grid(H, W, device):
        i, j = map(
            lambda x: x.float(),
            torch.meshgrid([torch.arange(0, H), torch.arange(0, W)], indexing="ij"),
        )
        return torch.stack([j, i], dim=-1).to(device)

    def deblur_tensor(self, raw_input, flow, mask=None):
        # raw: [B, V(bins), H, W]
        raw = raw_input.unsqueeze(2) if raw_input.ndim == 4 else raw_input
        N, T, C, H, W = raw.shape  # [B, V, 1, H, W]
        deblurred_tensor = torch.zeros_like(raw)
        identity_grid = self.create_identity_grid(H, W, raw.device)
        for t in range(T):
            delta_p = flow * t / (T - 1)  # reference coordinates prev frame
            # delta_p = flow * (T - 1 - t) / (T - 1)  # reference coordinates next frame
            sampling_grid = identity_grid + torch.movedim(
                delta_p, 1, -1
            )  # [B, H, W, 2]
            sampling_grid[..., 0] = sampling_grid[..., 0] / (W - 1) * 2 - 1
            sampling_grid[..., 1] = sampling_grid[..., 1] / (H - 1) * 2 - 1
            deblurred_tensor[
                :,
                t,
            ] = grid_sample(
                raw[
                    :,
                    t,
                ],
                sampling_grid,
                align_corners=False,
            )
        if raw_input.ndim == 4:
            deblurred_tensor = deblurred_tensor.squeeze(2)
        return deblurred_tensor  # [B, V(bins), H, W]

    def forward(self, event_bins, flow_init=None, deblur_iters=None, net_co=None):
        deblur_iters = self.deblur_iters
        x_raw = event_bins["event_volume"]

        B, V, H, W = x_raw.shape
        flow_total = (
            torch.zeros(B, 2, H, W).to(x_raw.device)
            if flow_init is None
            else flow_init.clone()
        )

        delta_flow = flow_total

        x_deblur = x_raw.clone()
        for iter in range(deblur_iters):
            x_deblur = self.deblur_tensor(x_deblur, delta_flow)
            x = torch.stack([x_deblur, x_deblur], dim=1)  # [B, 2, V(bins), H, W]
            if iter >= 1:
                net = self.cnet(flow_total)  # warm start module
            else:
                net = torch.zeros(
                    (
                        B,
                        self.hidden_dim,
                        H // self.downsample,
                        W // self.downsample,
                    )
                ).to(x.device)
            for i, slice in enumerate(
                x.permute(2, 0, 1, 3, 4)
            ):  # [V(bins), B, 2, H, W]
                f = self.fnet(slice)  # slice: [B, 2, H, W]
                net = self.update_net(net, f)

            dflow = self.update_net.compute_deltaflow(net)
            up_mask = self.update_net.compute_up_mask(net)
            delta_flow = self.upsample_flow(dflow, up_mask)

            flow_total = flow_total + delta_flow

        return {
            "final_prediction": flow_total,
            "net": net,
        }

    def forward_flowmap(self, event_bins, flow_init=None, deblur_iters=None):
        deblur_iters = self.deblur_iters if deblur_iters is None else deblur_iters

        x = event_bins["event_volume"]

        B, V, H, W = x.shape
        flow_total = (
            torch.zeros(B, 2, H, W).to(x.device)
            if flow_init is None
            else flow_init.clone()
        )

        delta_flow = flow_total

        for _ in range(deblur_iters):
            x_deblur = self.deblur_tensor(x, delta_flow)
            x = torch.stack([x_deblur, x_deblur], dim=1)

            if flow_init is not None and self.cnet is not None:
                net = self.cnet(flow_total)
            else:
                net = torch.zeros((B, self.hidden_dim, H // 8, W // 8)).to(x.device)
            for i, slice in enumerate(x.permute(2, 0, 1, 3, 4)):
                f = self.fnet(slice)
                net = self.update_net(net, f)

            dflow = self.update_net.compute_deltaflow(net)
            up_mask = self.update_net.compute_up_mask(net)
            delta_flow = self.upsample_flow(dflow, up_mask)
            flow_total = flow_total + delta_flow

        return {"final_prediction": flow_total}


class RecIDE(IDEDEQIDO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, batch, flow_init=None, deblur_iters=None):
        deblur_iters = self.deblur_iters if deblur_iters is None else deblur_iters

        flow_trajectory = []
        flow_next_trajectory = []

        for t, x in enumerate(batch):
            out = super().forward(x, flow_init=flow_init)
            flow_pred = out["final_prediction"]
            flow_init = self.forward_flow(flow_pred)

            flow_trajectory.append(flow_pred)

            if (t + 1) % 4 == 0:
                flow_init = flow_init.detach()
                yield {
                    "final_prediction": flow_pred,
                    "flow_trajectory": flow_trajectory,
                    "flow_next_trajectory": flow_next_trajectory,
                }
                flow_trajectory = []
                flow_next_trajectory = []

    def forward_inference(self, batch, flow_init=None, deblur_iters=None):
        deblur_iters = self.deblur_iters if deblur_iters is None else deblur_iters

        flow_trajectory = []

        for t, x in enumerate(batch):
            out = super().forward(x, flow_init=flow_init)
            flow_pred = out["final_prediction"]
            flow_init = self.forward_flow(flow_pred)
            flow_trajectory.append(flow_pred)

        return {"final_prediction": flow_pred, "flow_trajectory": flow_trajectory}

    def backward_neg_flow(self, x):
        x["event_volume"] = -torch.flip(x["event_volume"], [1])
        back_flow = -super().forward(x)["final_prediction"]
        return back_flow
