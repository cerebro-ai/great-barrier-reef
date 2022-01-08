""" Created by Dominik Schnaus at 12.12.2021.
Evaluation functions
"""
import json
import os
from typing import Tuple, Union, List

import wandb
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_iou
from torchvision.utils import draw_bounding_boxes
from wandb.sdk.data_types import WBValue

from gbr.data import GreatBarrierReefDataset, collate_fn, get_transform
from dataset.val_uploaded import image_ids_to_upload, img_id
from gbr.utils.tensorboard_utils import *
from gbr.utils.wandb_utils import MultipleVideoBuffer, create_box_data


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


def get_matched_indices(t):
    matched_row_indices = []
    matched_column_indices = []
    while True:
        row_indices = (t > 0).any(dim=1).nonzero().view(-1)
        if row_indices.shape[0] == 0:
            break
        row_index = row_indices[0].item()
        column_index = t[row_index].argmax().item()
        matched_row_indices.append(row_index)
        matched_column_indices.append(column_index)
        t[:, column_index] = 0
        t[row_index] = 0
    return torch.as_tensor(matched_row_indices, dtype=torch.long), \
           torch.as_tensor(matched_column_indices, dtype=torch.long)


def f_score(precision, recall, beta=2.):
    beta_2 = beta ** 2
    return (1 + beta_2) * (precision * recall) / (beta_2 * precision + recall)


class Evaluator:
    def __init__(self, device, iou_thresholds=None):
        self.device = device
        if iou_thresholds is None:
            iou_thresholds = torch.linspace(0.3, 0.8, int(np.round((0.8 - 0.3) / .05)) + 1)
        self.iou_thresholds = iou_thresholds
        self.mask_all = torch.zeros(0, iou_thresholds.shape[0], device=device)
        self.scores_all = torch.zeros(0, device=device)
        self.annotations_len_all = 0

    def update(self, predictions, scores, annotations):
        scores, indices = scores.sort(descending=True)
        predictions = predictions[indices]
        iou_values = box_iou(predictions, annotations)
        row_masks = torch.zeros(predictions.shape[0], self.iou_thresholds.shape[0], device=self.device)
        for iou_index, iou_threshold in enumerate(self.iou_thresholds):
            iou_values[iou_values < iou_threshold] = 0.
            prediction_indices, annotation_indices = get_matched_indices(iou_values.clone())
            row_masks[prediction_indices, iou_index] = True

        self.mask_all = torch.cat([self.mask_all, row_masks], dim=0)
        self.scores_all = torch.cat([self.scores_all, scores], dim=0)
        self.annotations_len_all += annotations.shape[0]

    def accumulate(self):
        self.scores_all, indices_all = self.scores_all.sort(descending=True)
        self.mask_all = self.mask_all[indices_all]
        cumsum = self.mask_all.cumsum(0)
        arange = torch.arange(1, self.mask_all.shape[0] + 1, device=self.device)
        precision = cumsum / arange[:, None]
        recall = cumsum / self.annotations_len_all
        f2_score = f_score(precision, recall)

        precision_average = precision.mean(dim=1)
        recall_average = recall.mean(dim=1)
        f2_score_average = f2_score.mean(dim=1)

        if f2_score_average.shape[0] > 0:
            f2_score_average.nan_to_num_(-1)
            f2_score_best, best_index = f2_score_average.max(dim=0)
            optimal_threshold = self.scores_all[best_index]
            precision_best = precision_average[best_index]
            recall_best = recall_average[best_index]

            metrics = {
                'precision': precision_best.item(),
                'recall': recall_best.item(),
                'F2 score': f2_score_best.item(),
                'optimal_threshold': optimal_threshold.item(),
            }
        else:
            metrics = {}
        return metrics, (recall_average, precision_average)


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             data_loader: torch.utils.data.DataLoader,
             device: torch.device) -> Tuple[Dict[str, float],
                                            Dict[int, Dict],
                                            Dict[str, List[wandb.Video]],
                                            Dict[str, WBValue]
]:
    """ Evaluates the model on the data_loader and device and returns the losses,
    COCO metrics and detections visualized on the image.

    Args:
        model: PyTorch model
        data_loader: PyTorch data loader
        device: device on which the model computations should be done

    Returns:
        measures: COCO metrics
        imgs_to_upload: image id -> {img, prediction, ground_truth}
        videos: List of wandb Videos
        wandb_objects: Dict {str -> WandB object}
    """
    model.eval()

    evaluator = Evaluator(device=device)
    # these set specifies image_id that will be uploaded to wandb with bounding boxes
    img_ids_to_upload = [img_id(x) for x in image_ids_to_upload]

    images_to_upload = {}
    multiple_video_buffer = MultipleVideoBuffer()

    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # torch.cuda.synchronize()
        outputs = model(images, targets)

        for image, output, target in zip(images, outputs, targets):
            image_id = target["image_id"].item()

            # check if it is an image which we want to upload
            image = (image * 255).to(device=torch.device('cpu'), dtype=torch.uint8)

            if image_id in img_ids_to_upload:
                images_to_upload[image_id] = {
                    "img": torch.permute(image, (1, 2, 0)).numpy(),  # permute such that (HxWxC)
                    "ground_truth": create_box_data(target, mode="ground_truth"),
                    "prediction": create_box_data(output, mode="prediction")
                }

            # draw image with boxes for video
            img_with_boxes = draw_bounding_boxes(image, target["boxes"], width=3, colors="red")
            img_with_boxes = draw_bounding_boxes(img_with_boxes, output["boxes"], width=2)

            # append to video buffer for the respective sub_sequence
            multiple_video_buffer.append(target["sub_sequence_id"].item(), img_with_boxes.numpy())

            evaluator.update(predictions=output['boxes'],
                             scores=output['scores'],
                             annotations=target['boxes'])

    # accumulate predictions from all images
    val_metrics, pr_curve = evaluator.accumulate()

    if val_metrics:
        data = [[x, y] for x, y in zip(*pr_curve)]
        table = wandb.Table(columns=["recall", "precision"], data=data)
        wandb_objects = {"pr_curve": wandb.plot_table(
            "wandb/area-under-curve/v0",
            table,
            {"x": "recall", "y": "precision"},
            {"title": "Precision v. Recall"},
        )}
    else:
        wandb_objects = {}

    videos_dict = multiple_video_buffer.export()
    multiple_video_buffer.reset()

    print(json.dumps(val_metrics, indent=4))
    return val_metrics, images_to_upload, videos_dict, wandb_objects


def evaluate_and_plot(model: torch.nn.Module,
                      data_loader: torch.utils.data.DataLoader,
                      device: torch.device,
                      epoch: int = 0,
                      ):
    val_metrics, images_to_upload, videos_dict, wandb_objects = evaluate(model, data_loader, device=device)

    videos_flattened = []
    for key, videos in videos_dict.items():
        for i, video in enumerate(videos):
            videos_flattened.append((f"{key}_{i}", video))

    # create log dict with images and videos
    wandb_log_dict = \
        {
            **{  # images
                str(image_id): wandb.Image(img_dict["img"], boxes={
                    "prediction": {
                        "box_data": img_dict["prediction"],
                        "class_labels": {1: "-", 2: "starfish"}
                    },
                    "ground_truth": {
                        "box_data": img_dict["ground_truth"],
                        "class_labels": {1: "starfish"}
                    }
                }) for image_id, img_dict in images_to_upload.items()},
            **{  # videos
                key: video for key, video in videos_flattened
            },
            # pr-curve and other plots
            **wandb_objects
        }

    return val_metrics, wandb_log_dict


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
