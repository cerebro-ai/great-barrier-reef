""" Created by Dominik Schnaus at 13.12.2021.
Main file to run the training.
"""
from pathlib import Path
from datetime import datetime

import wandb
import yaml

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from wandb.sdk.wandb_run import Run

from train import train_and_evaluate
from models.faster_rcnn import fasterrcnn_fpn


def get_latest_checkpoint(checkpoints_dir: Path):
    epoch = sorted([int(x.stem.split("_")[-1]) for x in checkpoints_dir.iterdir()])[-1]
    return checkpoints_dir.joinpath(f"Epoch_{epoch}.pth")


if __name__ == '__main__':
    # load config
    with Path("./config.yaml").open("r") as f:
        config = yaml.safe_load(f)

    params = config["params"]

    date_time = datetime.now().replace(microsecond=0).isoformat().replace(':', '_')
    run: Run = wandb.init(entity="cerebro-ai",
                          project="great-barrier-reef",
                          notes=f"{date_time}",
                          config=params
                          )

    run.summary["train_file"] = config["local"]["train_annotations_file"]
    run.summary["val_file"] = config["local"]["val_annotations_file"]

    model_name = config["params"]["model_name"]
    run.summary["model_name"] = model_name

    # get the checkpoint path from the run name and create the folder
    # fallback to date_time if wandb runs in offline mode
    # Path(checkpoints/eager-sunset-100)
    checkpoint_root = Path(config["local"]["checkpoint_root"]).joinpath(run.name if run.name else date_time)
    checkpoint_root.mkdir(exist_ok=True, parents=True)

    # if a checkpoint is given construct it from the checkpoint directory
    if run.resumed:
        existing_checkpoint_path = get_latest_checkpoint(checkpoint_root)
    else:
        existing_checkpoint = config["local"].get("resume_checkpoint", None)
        existing_checkpoint_path = str(checkpoint_root.parent.joinpath(existing_checkpoint)) \
            if existing_checkpoint \
            else None

    model = fasterrcnn_fpn(model_name, min_size_train=720, max_size_train=1280)
    # resnet50: train 512x512 -> batch size 20, val 720x1280 -> batch size 16
    # wide_resnet101_2: train 512x512 -> batch size 13, val 720x1280 -> batch size 16
    # efficientnet_b0: train 512x512 -> batch size 8, val 720x1280 -> batch size 6
    # efficientnet_b7: train 512x512 -> batch size 2, val 720x1280 -> batch size 1

    wandb.watch(model, log_freq=100)

    train_and_evaluate(
        model=model,
        root=config["local"]["dataset_root"],
        train_annotations_file=config["local"]["train_annotations_file"],
        val_annotations_file=config["local"]["val_annotations_file"],
        num_epochs=params["num_epochs"],
        checkpoint_path=str(checkpoint_root),
        eval_every_n_epochs=params["eval_every_n_epochs"],
        save_every_n_epochs=params["save_every_n_epochs"],
        train_batch_size=params["train_batch_size"],
        val_batch_size=params["val_batch_size"],
        train_num_workers=params["train_num_workers"],
        val_num_workers=params["val_num_workers"],
        learning_rate=float(params["learning_rate"]),
        existing_checkpoint_path=existing_checkpoint_path
    )
