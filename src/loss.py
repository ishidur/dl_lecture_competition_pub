import torch


def sparse_l1(estimated, ground_truth):
    return torch.mean(torch.abs(estimated - ground_truth))


def charbonnier_loss(delta, alpha=0.45, epsilon=1e-3):
    loss = torch.mean(torch.pow((delta**2 + epsilon**2), alpha))
    return loss


def compute_smoothness_loss(flow):
    flow_ucrop = flow[..., 1:]
    flow_dcrop = flow[..., :-1]
    flow_lcrop = flow[..., 1:, :]
    flow_rcrop = flow[..., :-1, :]

    flow_ulcrop = flow[..., 1:, 1:]
    flow_drcrop = flow[..., :-1, :-1]
    flow_dlcrop = flow[..., :-1, 1:]
    flow_urcrop = flow[..., 1:, :-1]

    smoothness_loss = (
        charbonnier_loss(flow_lcrop - flow_rcrop)
        + charbonnier_loss(flow_ucrop - flow_dcrop)
        + charbonnier_loss(flow_ulcrop - flow_drcrop)
        + charbonnier_loss(flow_dlcrop - flow_urcrop)
    )
    smoothness_loss /= 4.0

    return smoothness_loss


def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    """
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    """
    epe = torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1))
    return epe


def get_lossfunc(loss_conf):
    if loss_conf.name == "l1":
        return sparse_l1
    elif loss_conf.name == "l2":
        return compute_epe_error
    elif loss_conf.name == "l1+smooth":
        lmd = loss_conf.get("lambda", 0.5)
        print("lambda: ", lmd)

        def loss_fn(pred, gt):
            return sparse_l1(pred, gt) + lmd * compute_smoothness_loss(pred)

        return loss_fn
    elif loss_conf.name == "l2+smooth":
        lmd = loss_conf.get("lambda", 0.5)
        print("lambda: ", lmd)

        def loss_fn(pred, gt):
            return compute_epe_error(pred, gt) + lmd * compute_smoothness_loss(pred)

        return loss_fn
