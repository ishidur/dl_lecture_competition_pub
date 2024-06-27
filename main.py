import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from src.models.evflownet import EVFlowNet
from src.models.idedeq import IDEDEQIDO
from src.datasets import DatasetProvider
from src.datasets import train_collate
from src.loss import get_lossfunc, compute_epe_error
from src.utils import (
    RepresentationType,
    set_seed,
    move_batch_to_cuda,
)
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time
from logging import getLogger

logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    logger.info(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        """

    # ------------------
    #    Dataloader
    # ------------------
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=args.dataset.num_voxel_bins,
        config=args.dataset.train,
    )
    train_set = loader.get_train_dataset()
    # test_set = loader.get_test_dataset()
    val_set = loader.get_validation_dataset()
    collate_fn = train_collate
    train_data = DataLoader(
        train_set,
        batch_size=args.data_loader.train.batch_size,
        shuffle=args.data_loader.train.shuffle,
        num_workers=args.data_loader.train.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,
    )
    val_data = DataLoader(
        val_set,
        batch_size=args.data_loader.test.batch_size,
        shuffle=args.data_loader.test.shuffle,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,
    )

    """
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない
    
    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    """
    # ------------------
    #       Model
    # ------------------
    # model = EVFlowNet(args.model).to(device)
    model = IDEDEQIDO(args.model).to(device)
    model.load_state_dict(torch.load(args.model.pretrain_ckpt, map_location=device))

    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.train.initial_learning_rate,
        weight_decay=args.train.weight_decay,
    )
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=args.train.initial_learning_rate,
    # )
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        args.train.initial_learning_rate,
        epochs=args.train.epochs,
        steps_per_epoch=len(train_data),
    )
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
    loss_fn = get_lossfunc(args.loss)
    # ------------------
    #   Start training
    # # ------------------
    model.train()
    current_time = time.strftime("%Y%m%d%H%M%S")

    # Create the directory if it doesn't exist
    if not os.path.exists(f"checkpoints/{current_time}"):
        os.makedirs(
            f"checkpoints/{current_time}"
        )  # 71 EPE+smooth loss, 72 from scratch
    best_loss = None
    for epoch in range(args.train.epochs):
        total_loss = 0
        print("on epoch: {}".format(epoch + 1))
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]

            batch = move_batch_to_cuda(batch, device)
            out = model(batch)  # [B, 2, 480, 640]
            loss: torch.Tensor = loss_fn(out["final_prediction"], batch["flow_gt"])

            # event_image = batch["event_volume"].to(device)  # [B, 4, 480, 640]
            # ground_truth_flow = batch["flow_gt"].to(device)  # [B, 2, 480, 640]
            # flow = model(event_image)  # [B, 2, 480, 640]
            # loss: torch.Tensor = compute_epe_error(flow["flow3"], ground_truth_flow)
            # loss += 0.5*compute_smoothness_loss(flow["flow3"])

            # ground_truth_flow=F.avg_pool2d(ground_truth_flow, 2, 2)
            # loss += compute_epe_error(flow["flow2"], ground_truth_flow)
            # loss += 0.5*compute_smoothness_loss(flow["flow2"])

            # ground_truth_flow=F.avg_pool2d(ground_truth_flow, 2, 2)
            # loss += compute_epe_error(flow["flow1"], ground_truth_flow)
            # loss += 0.5*compute_smoothness_loss(flow["flow1"])

            # ground_truth_flow=F.avg_pool2d(ground_truth_flow, 2, 2)
            # loss += compute_epe_error(flow["flow0"], ground_truth_flow)
            # loss += 0.5*compute_smoothness_loss(flow["flow0"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        logger.info(
            # f"Epoch {epoch+1}, Loss: {total_loss / len(train_data)}"
            f"Epoch {epoch+1}, Loss: {total_loss / len(train_data)}, lr: {scheduler.get_last_lr()}"
        )
        model.eval()
        with torch.no_grad():
            print("start test")
            testloss = 0  # TODO: remove at final submission
            for batch in tqdm(val_data):
                batch: Dict[str, Any]
                batch = move_batch_to_cuda(batch, device)
                batch_flow = model(batch)  # [1, 2, 480, 640]
                loss: torch.Tensor = compute_epe_error(
                    batch_flow["final_prediction"], batch["flow_gt"]
                )  # TODO: remove at final submission

                # event_image = batch["event_volume"].to(device)
                # ground_truth_flow = batch["flow_gt"].to(device)  # [B, 2, 480, 640]
                # batch_flow = model(event_image)  # [1, 2, 480, 640]
                # loss: torch.Tensor = compute_epe_error(
                #     batch_flow["flow3"], ground_truth_flow
                # )  # TODO: remove at final submission
                testloss += loss.item()  # TODO: remove at final submission
            print("test done")
            if best_loss is None or best_loss > testloss:
                best_loss = testloss
                logger.info("Best score updated, save model")
                torch.save(
                    model.state_dict(), f"checkpoints/{current_time}/best_model.pth"
                )
            logger.info(
                f"TestLoss: {testloss / len(val_data)}, Best: {best_loss / len(val_data)}"
            )  # TODO: remove at final submission
        model.train()
        # scheduler.step(testloss)


if __name__ == "__main__":
    main()
