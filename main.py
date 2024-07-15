import os
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import DatasetProvider, train_collate
from src.loss import compute_epe_error, get_lossfunc
from src.models import get_eval_loss_fn, get_model, get_train_loss_fn
from src.utils import (
    RepresentationType,
    move_batch_to_cuda,
    set_seed,
)

logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
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
    patterns = [
        {"epochs": 100, "initial_learning_rate": 5e-4, "pct_start": 0.03, "weight_decay": None},
        {"epochs": 100, "initial_learning_rate": 1e-4, "pct_start": 0.05, "weight_decay": None},
        {"epochs": 100, "initial_learning_rate": 5e-4, "pct_start": 0.03, "weight_decay": 1e-4},
    ]
    for p in patterns:
        args.train.epochs = p["epochs"]
        args.train.initial_learning_rate = p["initial_learning_rate"]
        args.train.pct_start = p["pct_start"]
        args.train.weight_decay = p["weight_decay"]
        logger.info(args)
        # ------------------
        #       Model
        # ------------------
        model = get_model(args.model).to(device)
        if args.model.get("pretrain_ckpt", None):
            model.load_state_dict(torch.load(args.model.pretrain_ckpt, map_location=device))

        # ------------------
        #   optimizer
        # ------------------
        if args.train.get("weight_decay", None):
            logger.info(f"use AdamW, decay {args.train.weight_decay}")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.train.initial_learning_rate,
                weight_decay=args.train.weight_decay,
            )
        else:
            logger.info("use Adam")
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.train.initial_learning_rate,
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
            pct_start=args.train.pct_start,
            anneal_strategy="linear",
        )
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
        loss_fn = get_lossfunc(args.loss)
        train_loss = get_train_loss_fn(model, loss_fn)
        eval_loss = get_eval_loss_fn(model, compute_epe_error)
        # ------------------
        #   Start training
        # # ------------------
        model.train()
        current_time = time.strftime("%Y%m%d%H%M%S")

        # Create the directory if it doesn't exist
        if not os.path.exists(f"checkpoints/{current_time}"):
            os.makedirs(f"checkpoints/{current_time}")
        best_loss = None
        for epoch in range(args.train.epochs):
            total_loss = 0
            print("on epoch: {}".format(epoch + 1))
            for i, batch in enumerate(tqdm(train_data)):
                batch: Dict[str, Any]
                batch = move_batch_to_cuda(batch, device)
                loss = train_loss(model, batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}, Loss: {total_loss / len(train_data)}, lr: {scheduler.get_last_lr()}")
            model.eval()
            with torch.no_grad():
                print("start test")
                testloss = 0
                for batch in tqdm(val_data):
                    batch: Dict[str, Any] = move_batch_to_cuda(batch, device)
                    loss = eval_loss(model, batch)

                    testloss += loss.item()
                print("test done")
                if best_loss is None or best_loss > testloss:
                    best_loss = testloss
                    logger.info("Best score updated, save model")
                    torch.save(model.state_dict(), f"checkpoints/{current_time}/best_model.pth")
                logger.info(f"TestLoss: {testloss / len(val_data)}, Best: {best_loss / len(val_data)}")
            model.train()


if __name__ == "__main__":
    main()
