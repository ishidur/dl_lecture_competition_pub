from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import DatasetProvider, train_collate
from src.loss import compute_epe_error
from src.models import get_model
from src.utils import (
    RepresentationType,
    move_batch_to_cuda,
    save_optical_flow_to_npy,
    set_seed,
)


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
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    test_data = DataLoader(
        test_set,
        batch_size=args.data_loader.test.batch_size,
        shuffle=args.data_loader.test.shuffle,
        collate_fn=collate_fn,
        drop_last=False,
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
    model = get_model(args.model).to(device)
    # ------------------
    #   Start predicting
    # ------------------
    # model_path = f"idn/id-8x.pt"
    model_path = "checkpoints/20240703165626/best_model.pth"

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        total_loss = 0  # TODO: remove at final submission
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            ground_truth_flow = batch["flow_gt"].to(device)  # [B, 2, 480, 640]
            batch = move_batch_to_cuda(batch, device)

            # batch_flow = model(batch["event_volume"])  # [1, 2, 480, 640]
            # loss: torch.Tensor = compute_epe_error(
            #     batch_flow["flow3"], ground_truth_flow
            # )  # TODO: remove at final submission
            # flow = torch.cat((flow, batch_flow["flow3"]), dim=0)  # [N, 2, 480, 640]

            batch_flow = model(batch)  # [1, 2, 480, 640]
            loss: torch.Tensor = compute_epe_error(
                batch_flow["final_prediction"], ground_truth_flow
            )  # TODO: remove at final submission
            flow = torch.cat((flow, batch_flow["final_prediction"]), dim=0)  # [N, 2, 480, 640]
            total_loss += loss.item()  # TODO: remove at final submission
        print("test done")
        print(f"Loss: {total_loss / len(test_data)}")  # TODO: remove at final submission

    # ------------------
    #  save submission
    # ------------------
    file_name = "submission20240703165626"
    save_optical_flow_to_npy(flow, file_name)


if __name__ == "__main__":
    main()
