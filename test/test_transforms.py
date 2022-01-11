from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import ImageDraw
from torchvision.utils import draw_bounding_boxes

from gbr.data import GreatBarrierReefDataset, get_transform, collate_fn

# reproducibility
import random

random.seed(46)
torch.manual_seed(46)


def test_transformation():
    path = Path(__file__).parent.parent.joinpath("dataset").absolute()
    gbr_dataset = GreatBarrierReefDataset(root=str(path),
                                          annotation_path=str(Path(path).joinpath("reef_starter_0.05/train_clean.csv")),
                                          transforms=get_transform(True))
    data_loader_train = torch.utils.data.DataLoader(
        gbr_dataset, batch_size=3, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True)
    images, targets = next(iter(data_loader_train))
    images = (images * 255).to(device=torch.device('cpu'), dtype=torch.uint8)
    fig, axs = plt.subplots(3, 1, figsize=(7, 14))
    for i, image, target, ax in zip(range(images.shape[0]), images, targets, axs):
        print(i, target["image_id"])
        image = draw_bounding_boxes(image, target["boxes"], width=3, colors="red")
        ax.imshow(torch.permute(image, (1, 2, 0)).numpy())
    plt.show()


if __name__ == '__main__':
    test_transformation()
