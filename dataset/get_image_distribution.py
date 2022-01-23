from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gbr.data.gbr_dataset import GreatBarrierReefDataset, get_transform, collate_fn
import yaml

with Path(__file__).parent.parent.joinpath("config.yaml").open("r") as f:
    config = yaml.safe_load(f)

dataset = GreatBarrierReefDataset(
    root=config["local"]["dataset_root"],
    annotation_path=str(Path(config["local"]["dataset_root"]).joinpath("all.csv")),
    transforms=get_transform(train=False)
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=20,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

mean_image = torch.zeros((3, 720, 1280))

for images, _ in tqdm(dataloader):
    sum_image = torch.sum(images, dim=0)
    mean_image += sum_image


mean_image /= len(dataset.annotation_file) - 1

rgb_mean = torch.mean(mean_image, (1, 2))
rgb_std = torch.std(mean_image, (1, 2))

print("Mean", rgb_mean)
print("Std", rgb_std)

# Result on all.csv
# Mean tensor([0.2652, 0.5725, 0.6195])
# Std tensor([0.0585, 0.0278, 0.0244])
