""" Created by Dominik Schnaus at 13.12.2021.
Main file to run the training.
"""
from pathlib import Path
from datetime import datetime
import yaml

from torchvision.models.detection import fasterrcnn_resnet50_fpn

from train import train_and_evaluate

if __name__ == '__main__':
    # load config
    with Path("./config.yaml").open("r") as f:
        config = yaml.safe_load(f)

    params = config["params"]

    date_time = datetime.now().replace(microsecond=0).isoformat().replace(':', '_')
    # TODO get checkpoint root from config or sane default and not dominiks path ;)
    checkpoint_root = Path(config["local"]["checkpoint_root"])

    checkpoint_root = checkpoint_root.joinpath(date_time)
    checkpoint_root.mkdir(exist_ok=True, parents=True)

    model = fasterrcnn_resnet50_fpn(num_classes=2,
                                    trainable_backbone_layers=5,
                                    image_mean=[0.2652, 0.5724, 0.6195],
                                    image_std=[0.2155, 0.1917, 0.1979])

    train_and_evaluate(
        model=model,
        root=config["local"]["dataset_root"],
        num_epochs=params["num_epochs"],
        checkpoint_path=str(checkpoint_root),
        eval_every_n_epochs=params["eval_every_n_epochs"],
        save_every_n_epochs=params["save_every_n_epochs"],
        train_batch_size=params["train_batch_size"],
        val_batch_size=params["val_batch_size"],
        train_num_workers=params["train_num_workers"],
        val_num_workers=params["val_num_workers"],
        learning_rate=float(params["learning_rate"]),
    )
