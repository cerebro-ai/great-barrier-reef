""" Created by Dominik Schnaus at 12.12.2021.
Data augmentations for object detection
"""

from torchvision.transforms import functional as F


class Compose:
    """ Composes multiple transformations to one transformation.

    Applies the transforms in the given order on the input.

    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    """ Transforms the PIL Image to a tensor."""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
