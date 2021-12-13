""" Created by Dominik Schnaus at 12.12.2021.
Evaluation functions
"""
import os
from typing import Tuple, Union, List

import wandb
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import draw_bounding_boxes

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from data import GreatBarrierReefDataset, collate_fn, get_transform
from tensorboard_utils import *


def reduce_dict(dict_list: List[Dict[str, float]],
                reduction: str) -> Dict[str, Union[np.ndarray, float]]:
    """ Reduces the values from multiple dicts to one dict containing the
    averages for every key.

    Args:
        dict_list: list of dicts, all having the same keys
        reduction: how the dict should be reduced, possible_values: 'sum',
            'mean'

    Returns:
        dict with same keys and reduced values
    """
    reduction_dict = {
        'sum': np.sum,
        'mean': np.mean,
    }
    reduction_fn = reduction_dict[reduction]
    out = {}
    if len(dict_list) > 0:
        for key in dict_list[0].keys():
            out[key] = reduction_fn([d[key] for d in dict_list], axis=0)
    return out


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             data_loader: torch.utils.data.DataLoader,
             device: torch.device) -> Tuple[Dict[str, float],
                                            Dict[int, np.array]]:
    """ Evaluates the model on the data_loader and device and returns the losses,
    COCO metrics and detections visualized on the image.

    Args:
        model: PyTorch model
        data_loader: PyTorch data loader
        device: device on which the model computations should be done

    Returns:
        measures: COCO metrics
        results: image id -> image with bounding boxes
    """
    stats_tags = ['Precision/mAP', 'Precision/mAP@.50IOU', 'Precision/mAP@.75IOU', 'Precision/mAP (small)',
                  'Precision/mAP (medium)', 'Precision/mAP (large)',
                  'Recall/AR@1', 'Recall/AR@10', 'Recall/AR@100', 'Recall/AR@100 (small)', 'Recall/AR@100 (medium)',
                  'Recall/AR@100 (large)']
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"])

    results = {}
    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # torch.cuda.synchronize()
        outputs = model(images, targets)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        imgs_with_boxes = []
        for image, output, target in zip(images, outputs, targets):
            image = (image * 255).to(device=cpu_device, dtype=torch.uint8)
            image = draw_bounding_boxes(image, target['boxes'], width=3, colors='red')
            image = draw_bounding_boxes(image, output['boxes'], width=3)
            imgs_with_boxes.append(image)
        results.update({target["image_id"].item(): img for target, img in zip(targets, imgs_with_boxes)})

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = coco_evaluator.coco_eval['bbox'].stats
    val_metrics = {tag: stat for tag, stat in zip(stats_tags, stats)}

    torch.set_num_threads(n_threads)
    return val_metrics, results


def evaluate_and_plot(model: torch.nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      device: torch.device,
                      epoch: int = 0,
                      ):
    val_metrics, results = \
        evaluate(model, data_loader, device=device)

    wandb.log({
        "Validation "+key: value for
        key, value in val_metrics.items()
    })

    if results:
        for key, image in results.values():
            wandb.log({str(key): wandb.Image(image)})

    return val_metrics, results


def evaluate_path(model: torch.nn.Module,
                  root: str,
                  test_file: str,
                  checkpoint_path: str,
                  test_batch_size=6,
                  test_num_workers=6):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset_test = GreatBarrierReefDataset(root=root, annotation_file=test_file,
                                           transforms=get_transform(False))
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=test_batch_size, shuffle=False,
        num_workers=test_num_workers, pin_memory=True, collate_fn=collate_fn)
    writer = SummaryWriter(os.path.join(checkpoint_path, 'Tensorboard'))
    return evaluate_and_plot(model, data_loader_test, writer, device=device,
                             epoch=0)
