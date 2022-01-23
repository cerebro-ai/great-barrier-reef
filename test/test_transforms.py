from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

from gbr.data.augmentations import draw_torch_image
from gbr.data.gbr_dataset import GreatBarrierReefDataset, get_transform, collate_fn

# reproducibility
import random


# random.seed(47)
# torch.manual_seed(47)


def test_transformation():
    path = Path(__file__).parent.parent.joinpath("dataset").absolute()
    gbr_dataset = GreatBarrierReefDataset(root=str(path),
                                          annotation_path=str(Path(path).joinpath("reef_starter_0.05/train_clean.csv")),
                                          transforms=get_transform(True),
                                          copy_paste=False,
                                          apply_mixup=True,
                                          min_side_length=40)
    data_loader_train = torch.utils.data.DataLoader(
        gbr_dataset, batch_size=3, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True)
    images, targets = next(iter(data_loader_train))
    images = (images * 255).to(device=torch.device('cpu'), dtype=torch.uint8)
    fig, axs = plt.subplots(3, 1, figsize=(7, 14))
    for i, image, target, ax in zip(range(images.shape[0]), images, targets, axs):
        print(i, target["image_id"], image.shape)
        image = draw_bounding_boxes(image, target["boxes"], width=2, colors="red")
        ax.imshow(torch.permute(image, (1, 2, 0)).numpy())
    plt.show()


def test_bboxes():
    path = Path(__file__).absolute().parent.parent.joinpath("dataset")
    print(path)
    gbr_dataset = GreatBarrierReefDataset(root=str(path),
                                          annotation_path=str(Path(path).joinpath("reef_starter_0.05/train_clean.csv")),
                                          transforms=get_transform(True),
                                          copy_paste=True,
                                          apply_mixup=False)

    data_loader_train = torch.utils.data.DataLoader(
        gbr_dataset, batch_size=4, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=True)

    for i, (images, targets) in tqdm(enumerate(data_loader_train), total=len(data_loader_train)):
        for image, target in zip(images, targets):
            bboxes = target["boxes"]
            wh = bboxes[:, 2:] - bboxes[:, :2]
            if torch.any(wh < 25):
                draw_torch_image(image, target["boxes"])
                input("holding...")


if __name__ == '__main__':
    test_bboxes()
