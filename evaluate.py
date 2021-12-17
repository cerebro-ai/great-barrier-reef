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
from dataset.val_uploaded import image_ids_to_upload, img_id
from tensorboard_utils import *
from wandb_utils import VideoBuffer, create_box_data


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
                                            Dict[int, Dict],
                                            List[wandb.Video]
]:
    """ Evaluates the model on the data_loader and device and returns the losses,
    COCO metrics and images to be uploaded with their prediction and target boxes, and a list of videos.

    Args:
        model: PyTorch model
        data_loader: PyTorch data loader
        device: device on which the model computations should be done

    Returns:
        measures: COCO metrics
        imgs_to_upload: image id -> {img, prediction, ground_truth}
        videos: List of wandb Videos
    """
    stats_tags = ['Precision/mAP', 'Precision/mAP@.50IOU', 'Precision/mAP@.75IOU', 'Precision/mAP (small)',
                  'Precision/mAP (medium)', 'Precision/mAP (large)',
                  'Recall/AR@1', 'Recall/AR@10', 'Recall/AR@100', 'Recall/AR@100 (small)', 'Recall/AR@100 (medium)',
                  'Recall/AR@100 (large)']
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()

    # these set specifies image_id that will be uploaded to wandb with bounding boxes
    img_ids_to_upload = [img_id(x) for x in image_ids_to_upload]

    coco = get_coco_api_from_dataset(data_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"])

    images_to_upload = {}
    video_buffer = VideoBuffer()

    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # torch.cuda.synchronize()
        outputs = model(images, targets)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        for image, output, target in zip(images, outputs, targets):
            image_id = target["image_id"].item()

            # check if it is an image which which we want to upload
            image = (image * 255).to(device=cpu_device, dtype=torch.uint8)

            if image_id in img_ids_to_upload:
                images_to_upload[image_id] = {
                    "img": torch.permute(image, (1, 2, 0)).numpy(),  # permute such that (HxWxC)
                    "ground_truth": create_box_data(target, mode="ground_truth"),
                    "prediction": create_box_data(output, mode="prediction")
                }

            # draw image with boxes for video
            img_with_boxes = draw_bounding_boxes(image, target["boxes"], width=3, colors="red")
            img_with_boxes = draw_bounding_boxes(img_with_boxes, output["boxes"], width=2)
            # append to video buffer
            video_buffer.append(img_with_boxes.numpy())

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = coco_evaluator.coco_eval['bbox'].stats
    val_metrics = {tag: stat for tag, stat in zip(stats_tags, stats)}

    videos = video_buffer.export()
    video_buffer.reset()

    torch.set_num_threads(n_threads)
    return val_metrics, images_to_upload, videos


def evaluate_and_plot(model: torch.nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      device: torch.device,
                      epoch: int = 0,
                      ):
    val_metrics, images_to_upload, videos = evaluate(model, data_loader, device=device)

    wandb.log({
        "Validation " + key: value for
        key, value in val_metrics.items()
    })

    for image_id, img_dict in images_to_upload.items():
        wandb.log({str(image_id): wandb.Image(img_dict["img"], boxes={
            "prediction": {
                "box_data": img_dict["prediction"],
                "class_labels": {1: "-", 2: "starfish"}
            },
            "ground_truth": {
                "box_data": img_dict["ground_truth"],
                "class_labels": {1: "starfish"}
            }
        })})

    for i, video in enumerate(videos):
        wandb.log({f"video_{i}": video})

    return val_metrics


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
