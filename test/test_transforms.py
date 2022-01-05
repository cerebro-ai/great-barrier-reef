from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import ImageDraw
from torchvision.utils import draw_bounding_boxes

from data import GreatBarrierReefDataset, get_transform, collate_fn


def test_transformation():
    path = Path(__file__).parent.parent.joinpath("dataset").absolute()
    gbr_dataset = GreatBarrierReefDataset(root=str(path), annotation_file=str("sub_sequence_split/val.csv"),
                                          transforms=get_transform(True))
    data_loader_train = torch.utils.data.DataLoader(
        gbr_dataset, batch_size=4, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True)
    images, targets = next(iter(data_loader_train))
    images = (images * 255).to(device=torch.device('cpu'), dtype=torch.uint8)
    fig, axs = plt.subplots(1, 4, figsize=(30, 5))
    for i, image, target, ax in zip(range(images.shape[0]), images, targets, axs):
        print(i, target["image_id"])
        image = draw_bounding_boxes(image, target["boxes"], width=1, colors="red")
        ax.imshow(torch.permute(image, (1, 2, 0)).numpy())
    plt.show()


if __name__ == '__main__':
    test_transformation()