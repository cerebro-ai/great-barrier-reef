""" Created by Dominik Schnaus at 11.12.2021.
Dataset for the great barrier reef kaggle challenge
(https://www.kaggle.com/c/tensorflow-great-barrier-reef/data)
"""
import logging
import random
from os.path import join
from typing import Dict, Tuple, Union, Any, List

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations import DualTransform
from albumentations.pytorch import transforms as At
from albumentations.augmentations.crops import functional as F

from gbr.data.augmentations import copy_paste_augmentation


class GreatBarrierReefDataset(torch.utils.data.Dataset):
    """GreatBarrierReefDataset.

    Attributes:
        image_root: path to the folder with the images
        annotation_file: file with the annotations (train.csv or val.csv)
        transforms: transformations that should be applied on the images and targets.
            See https://albumentations.ai/docs/

    """

    ANNOTATIONS_COLUMN = "clamped_annotations"

    def __init__(self,
                 root: str,
                 annotation_path: str,
                 transforms=None,
                 copy_paste=False,
                 apply_mixup=False,
                 min_side_length=None
                 ):
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

        self.annotation_file = pd.read_csv(annotation_path)
        self.annotation_file[self.ANNOTATIONS_COLUMN] = self.annotation_file[self.ANNOTATIONS_COLUMN].apply(eval)
        self.copy_paste = copy_paste
        self.apply_mixup = apply_mixup
        self.transforms = transforms
        self.min_side_length = min_side_length

    def pull_item(self, idx: int):
        """Return an item and apply the per image augmentations

        Args:
            idx: Index

        Returns (tuple): Image, Target
        """
        annotations = self.annotation_file.loc[idx]
        boxes = torch.tensor([list(box.values()) for box in annotations[self.ANNOTATIONS_COLUMN]]).view(-1, 4)
        boxes[:, 2:] += boxes[:, :2]

        image_path = join(self.image_root, f'video_{annotations.video_id}', f'{annotations.video_frame}.jpg')
        image = np.asarray(Image.open(image_path))

        meta_keys = ['video_id', 'sequence', 'video_frame', 'sequence_frame']

        # needed for pycocotools
        image_id = f'{annotations.video_id}{annotations.video_frame:05d}'
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros(boxes.shape[0], dtype=torch.int64)
        labels = np.zeros(boxes.shape[0])

        target = {'boxes': boxes,
                  'labels': torch.zeros(boxes.shape[0], dtype=torch.int64),
                  'image_id': torch.as_tensor(int(image_id)),
                  'area': area,
                  'iscrowd': iscrowd,
                  **{key: torch.as_tensor(value) for key, value in annotations[meta_keys].items()}}

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=target["boxes"], labels=labels)
            image = transformed["image"] / 255
            target["boxes"] = torch.tensor(transformed["bboxes"]).view(-1, 4)

            # filter out all boxes that have a side that is smaller than <min_side_length> pixels
            if self.min_side_length is not None:
                target["boxes"] = filter_boxes(
                    lambda box: torch.minimum(box[2] - box[0], box[3] - box[1]) > self.min_side_length,
                    target["boxes"])

            target["area"] = (target["boxes"][:, 2] - target["boxes"][:, 0]) * (
                    target["boxes"][:, 3] - target["boxes"][:, 1])

        # update labels if some boxes could deleted during the augmentation process
        target["labels"] = torch.zeros(target["boxes"].shape[0], dtype=torch.int64)

        return image, target

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
        image1, target1 = self.pull_item(idx)

        if self.copy_paste:
            rand_idx = random.randint(0, len(self.annotation_file) - 1)
            image2, target2 = self.pull_item(rand_idx)
            image, boxes = copy_paste_augmentation(image1, target1["boxes"], image2, target2["boxes"])
            if boxes.shape[0] == 0:
                rand_idx = random.randint(0, len(self.annotation_file) - 1)
                image2, target2 = self.pull_item(rand_idx)
                image, boxes = copy_paste_augmentation(image1, target1["boxes"], image2, target2["boxes"])
            target = update_target_boxes(target1, boxes)


        elif self.apply_mixup:
            raise NotImplementedError
            # rand_idx = random.randint(0, len(self.annotation_file))
            # image2, target2 = self.pull_item(rand_idx)
            # image = 0.5 * image1 + 0.5 * image2
            # target = combine_targets(target1, target2)
        else:
            image, target = image1, target1

        return image, target

    def __len__(self) -> int:
        """ Number of elements in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.annotation_file)


def filter_boxes(f, input_boxes):
    output = [box for box in input_boxes if f(box)]
    if len(output) == 0:
        return torch.tensor([]).view(0, 4)
    else:
        return torch.vstack(output).view(-1, 4)


def l1_dist(box, boxes):
    return torch.max(torch.abs(boxes - box), dim=1).values


def center(boxes: torch.Tensor):
    if boxes.dim() == 1:
        return boxes[0:2] + 0.5 * boxes[2:]
    return boxes[:, 0:2] + 0.5 * boxes[:, 2:]


def update_target_boxes(t1, boxes):
    """Combines the targets t1 and t2

    Args:
        idxs: Specify a subset of t2 that should be combined

    Returns:

    """
    area = (boxes[:, 2] - boxes[:, 0]) * (
            boxes[:, 3] - boxes[:, 1])
    target = t1
    target["boxes"] = boxes
    target["labels"] = torch.zeros(boxes.shape[0], dtype=torch.int64)
    target["area"]: area
    target["iscrowd"]: torch.zeros(boxes.shape[0], dtype=torch.int64)
    return target


def collate_fn(batch):
    c = list(zip(*batch))
    return torch.stack(c[0]), c[1]


class RandomCropAroundRandomBox(DualTransform):
    """Choose a random crop such that it contains at least on bounding box.

    Parts outside the image are extrapolated with 'black' are by mirroring the image

    """

    def __init__(self, height, width, pad_mode=cv2.BORDER_DEFAULT, always_apply=False, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.height = height
        self.width = width
        self.pad_mode = pad_mode

        # top bottom left right
        self.pad_params = [self.height, self.height, self.width, self.width]

    def apply(self, img, x_min, x_max, y_min, y_max, **params):
        # pad the img
        interpolation = cv2.INTER_LINEAR
        h, w, _ = img.shape
        img = F.crop_and_pad(img, None, self.pad_params, 0, None, None, interpolation, cv2.BORDER_CONSTANT, False)
        h_b, w_b, _ = img.shape
        # account for the padding
        x_min = int(self.width + x_min * w)
        x_max = int(self.width + x_max * w)
        y_min = int(self.height + y_min * h)
        y_max = int(self.height + y_max * h)

        return F.crop(img, x_min, y_min, x_max, y_max)

    def apply_to_bbox(self, bbox, x_min, x_max, y_min, y_max, rows, cols, **params):
        # img_h, img_w, _ = params["image"].shape
        result_rows = rows + 2 * self.pad_params[0]
        result_cols = cols + 2 * self.pad_params[2]
        bbox = F.crop_and_pad_bbox(bbox,
                                   crop_params=None,
                                   pad_params=self.pad_params,
                                   rows=rows,
                                   cols=cols,
                                   result_rows=result_rows,
                                   result_cols=result_cols,
                                   keep_size=True)
        x_min = int(self.width + x_min * cols)
        x_max = int(self.width + x_max * cols)
        y_min = int(self.height + y_min * rows)
        y_max = int(self.height + y_max * rows)
        return F.bbox_crop(bbox, x_min, y_min, x_max, y_max, result_rows, result_cols)

    def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, int]:
        n_boxes = len(params["bboxes"])
        img_h, img_w, _ = params["image"].shape
        if n_boxes == 0:
            # no boxes return random crop inside image
            crop_x = random.uniform(0, (img_w - self.width) / img_w)
            crop_y = random.uniform(0, (img_h - self.height) / img_h)
            return {"x_min": crop_x,
                    "x_max": crop_x + self.width / img_w,
                    "y_min": crop_y,
                    "y_max": crop_y + self.height / img_h,
                    "rows": img_h,
                    "cols": img_w
                    }
        if n_boxes == 1:
            bbox = params["bboxes"][0]
        else:
            # choose random box
            bbox = random.choice(params["bboxes"])
        min_x = bbox[2] - (3 / 4 * self.width / img_w)
        max_x = bbox[0] - (1 / 4 * self.width / img_w)

        min_y = bbox[3] - (3 / 4 * self.height / img_h)
        max_y = bbox[1] - (1 / 4 * self.height / img_h)

        crop_x = random.uniform(min_x, max_x)
        crop_y = random.uniform(min_y, max_y)

        return {"x_min": crop_x,
                "x_max": crop_x + self.width / img_w,
                "y_min": crop_y,
                "y_max": crop_y + self.height / img_h,
                "rows": img_h,
                "cols": img_w
                }

    @property
    def targets_as_params(self) -> List[str]:
        return ["image", "bboxes"]

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "height", "width"


def get_transform(train: bool = True,
                  **config_params
                  ) -> A.Compose:
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
    # no default

    rotation_limit = int(config_params.get("rotation_limit", 0))
    random_scale = config_params.get("random_scale", (0.8, 1.05))
    random_rain_prob = float(config_params.get("random_rain_prob", 0))

    if train:
        h, w = config_params.get("input_size", (512, 512))
        transforms = [
            A.Rotate(rotation_limit, border_mode=cv2.BORDER_CONSTANT, p=1),
            A.Perspective(p=1),
            A.HorizontalFlip(p=0.5),
            A.RandomScale(scale_limit=random_scale, p=1),
            RandomCropAroundRandomBox(h, w),
            A.Resize(height=h, width=w),
            # A.RGBShift(p=1),
            # A.Equalize(p=1),
            # A.ColorJitter(brightness=0.05, hue=0.05, contrast=0.05, saturation=0.05, p=1),
            A.RandomRain(p=random_rain_prob)
        ]
    else:
        h, w = config_params.get("test_size", (720, 1280))
        transforms = [
            A.Resize(height=h, width=w)
            # YOlOX breaks with the original (720, 1280) because they are not dividable with the highest stride 32
            # RandomCropAroundRandomBox(hw, hw)  # TODO remove this
        ]
    transforms.append(At.ToTensorV2())
    return A.Compose(transforms, bbox_params=A.BboxParams(format="pascal_voc",
                                                          label_fields=["labels"],
                                                          min_visibility=0.2
                                                          )
                     )
