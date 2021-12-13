""" Created by Dominik Schnaus at 13.12.2021.
Main file to run the training.
"""
import os
from datetime import datetime

from torchvision.models.detection import fasterrcnn_resnet50_fpn

from train import train_and_evaluate


if __name__ == '__main__':
    date_time = datetime.now().replace(microsecond=0).isoformat().replace(
        ':', '_')
    checkpoint_root = '/home/dominik/PycharmProjects/great-barrier-reef/checkpoints'
    checkpoint_root = os.path.join(checkpoint_root, date_time)
    os.mkdir(checkpoint_root)

    model = fasterrcnn_resnet50_fpn(num_classes=2,
                                    trainable_backbone_layers=5,
                                    image_mean=[0.2652, 0.5724, 0.6195],
                                    image_std=[0.2155, 0.1917, 0.1979])

    train_and_evaluate(
        model=model,
        root='/home/dominik/PycharmProjects/great-barrier-reef/dataset',
        num_epochs=100,
        checkpoint_path=checkpoint_root,
        eval_every_n_epochs=1,
        save_every_n_epochs=5,
        train_batch_size=6,
        val_batch_size=6,
        train_num_workers=0,
        val_num_workers=0,
        learning_rate=1e-4,
    )
