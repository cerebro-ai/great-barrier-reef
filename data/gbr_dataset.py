""" Created by Dominik Schnaus at 11.12.2021.
Dataset for the great barrier reef kaggle challenge
(https://www.kaggle.com/c/tensorflow-great-barrier-reef/data)
"""

from os.path import join
from typing import Dict, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations.pytorch import transforms as At


class GreatBarrierReefDataset(torch.utils.data.Dataset):
    """GreatBarrierReefDataset.

    Attributes:
        image_root: path to the folder with the images
        annotation_file: file with the annotations (train.csv or val.csv)
        transforms: transformations that should be applied on the images and targets.
            See https://albumentations.ai/docs/

    """

    def __init__(self,
                 root: str,
                 annotation_file: str,
                 transforms: A.DualTransform = None):
        """ Inits the great barrier reef dataset.

        The root path should contain the subfolder train_images with subfolders
        video_0, video_1, ...and the file train.csv for the annotations. The
        images should end with '.jpg'

        Args:
            root: path to the root folder containing the images in the subfolder
                'train_images/video_' and the annotations in train.csv
            transforms: transformation which takes as an input a PIL image and
                a dict with keys 'boxes' and meta data keys and returns a
                tensor and a dict with the same keys. (optional)
        """
        self.image_root = join(root, 'train_images')
        annotation_path = join(root, annotation_file)

        self.annotation_file = pd.read_csv(annotation_path)
        self.annotation_file.annotations = self.annotation_file.annotations.apply(eval)

        self.transforms = transforms

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor,
                                                   Dict[str, torch.Tensor]],
                                             Tuple[Image.Image,
                                                   Dict[str, torch.Tensor]]]:
        """ Returns a transformed image and corresponding annotations.

        Args:
            idx: index which image and annotation will be returned

        Returns:
            a tensor (or a PIL Image if transform is None) and a dict with keys
                'annotations' and 'image_id' (output of transform else without
                transform)
        """
        annotations = self.annotation_file.loc[idx]
        boxes = np.array([list(box.values()) for box in annotations.annotations])
        if boxes.shape[0] != 0:
            boxes[:, 2:] += boxes[:, :2]
        else:
            boxes = boxes.view(0, 4)

        image_path = join(self.image_root, f'video_{annotations.video_id}', f'{annotations.video_frame}.jpg')
        image = np.asarray(Image.open(image_path))

        meta_keys = ['video_id', 'sequence', 'video_frame', 'sequence_frame']

        # needed for pycocotools
        image_id = f'{annotations.video_id}{annotations.video_frame:05d}'
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros(boxes.shape[0], dtype=torch.int64)
        labels = np.zeros(boxes.shape[0])

        target = {'boxes': boxes, 'labels': torch.ones(boxes.shape[0], dtype=torch.int64),
                  'image_id': torch.as_tensor(int(image_id)), 'area': area, 'iscrowd': iscrowd,
                  **{key: torch.as_tensor(value) for key, value in annotations[meta_keys].items()}}

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=target["boxes"], labels=labels)
            image = transformed["image"] / 255
            target["boxes"] = torch.tensor(transformed["bboxes"])
            target["area"] = (target["boxes"][:, 2] - target["boxes"][:, 0]) * (
                    target["boxes"][:, 3] - target["boxes"][:, 1])

        return image, target

    def __len__(self) -> int:
        """ Number of elements in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.annotation_file)


def collate_fn(batch):
    c = list(zip(*batch))
    return torch.stack(c[0]), c[1]


def get_transform(train: bool = True,
                  size: Tuple[int, int] = (512, 512)) -> A.Compose:
    """ Returns the transforms for the images and targets.

    Transformations:
        eval: ToTensor()
        train: TODO: Which transformations should be done?
    Args:
        train: True iff model is training. (More augmentations are done)
        size: (shape (H, W)) size of output image.

    Returns:
        callable that applies the transformations on images and targets.
    """
    if train:
        transforms = [
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomSizedBBoxSafeCrop(height=720, width=1280),
                A.ShiftScaleRotate(),
            ]),
            A.RandomBrightnessContrast(p=0.5),
            A.MotionBlur(),
        ]
    else:
        transforms = []
    transforms.append(At.ToTensorV2())
    return A.Compose(transforms, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]))
