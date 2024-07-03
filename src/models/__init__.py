from .evflownet import EVFlowNet
from .idedeq import IDEDEQIDO
from torch.nn import functional as F
import torch


def get_model(config):
    model_name = config.get("name", None)
    if model_name == "evflownet":
        return EVFlowNet(config)
    elif model_name == "idnet":
        return IDEDEQIDO(config)
    else:
        raise ValueError(f"Model not found: {model_name}")


def get_train_loss_fn(model, loss_fn):
    if isinstance(model, EVFlowNet):

        def tloss(_model, batch):
            event_image = batch["event_volume"]  # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"]  # [B, 2, 480, 640]
            flow = _model(event_image)  # [B, 2, 480, 640]
            loss: torch.Tensor = loss_fn(flow["flow3"], ground_truth_flow)

            ground_truth_flow = F.avg_pool2d(ground_truth_flow, 2, 2)
            loss += loss_fn(flow["flow2"], ground_truth_flow)

            ground_truth_flow = F.avg_pool2d(ground_truth_flow, 2, 2)
            loss += loss_fn(flow["flow1"], ground_truth_flow)

            ground_truth_flow = F.avg_pool2d(ground_truth_flow, 2, 2)
            loss += loss_fn(flow["flow0"], ground_truth_flow)
            return loss

        return tloss
    elif isinstance(model, IDEDEQIDO):

        def tloss(_model, batch):
            out = _model(batch)  # [B, 2, 480, 640]
            loss: torch.Tensor = loss_fn(out["final_prediction"], batch["flow_gt"])
            return loss

        return tloss


def get_eval_loss_fn(model, loss_fn):
    if isinstance(model, EVFlowNet):

        def eloss(_model, batch):
            event_image = batch["event_volume"]  # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"]  # [B, 2, 480, 640]
            flow = _model(event_image)  # [B, 2, 480, 640]
            loss: torch.Tensor = loss_fn(flow["flow3"], ground_truth_flow)
            return loss

        return eloss
    elif isinstance(model, IDEDEQIDO):

        def eloss(_model, batch):
            out = _model(batch)  # [B, 2, 480, 640]
            loss: torch.Tensor = loss_fn(out["final_prediction"], batch["flow_gt"])
            return loss

        return eloss
