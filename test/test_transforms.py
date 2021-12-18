from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes

from data import GreatBarrierReefDataset, get_transform


def test_transformation():
    path = Path(__file__).parent.parent.joinpath("dataset")
    annotation_file = Path(__file__).parent.parent.joinpath("dataset/all.csv")
    gbr = GreatBarrierReefDataset(root=str(path), annotation_file=str(annotation_file), transforms=get_transform(True))
    image, target = gbr[100]
    image = (image * 255).to(device=torch.device('cpu'), dtype=torch.uint8)
    image = draw_bounding_boxes(image, target["boxes"], width=5, colors="red")
    plt.imshow(torch.permute(image, (1, 2, 0)).numpy())
    plt.show()
