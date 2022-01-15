""" Created by Dominik Schnaus at 12.12.2021.
Training file for the great barrier reef kaggle challenge
(https://www.kaggle.com/c/tensorflow-great-barrier-reef/data)
"""
import math
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List

import wandb
from torch.utils.tensorboard import SummaryWriter

from gbr.data.gbr_dataset import GreatBarrierReefDataset, collate_fn, get_transform
from gbr.evaluate import evaluate_and_plot
from gbr.utils.tensorboard_utils import *


def get_latest_checkpoint(checkpoints_dir: Path):
    epoch = sorted([int(x.stem.split("_")[-1]) for x in checkpoints_dir.iterdir()])[-1]
    return checkpoints_dir.joinpath(f"Epoch_{epoch}.pth")


def get_data_loaders(root: str,
                     train_annotations_file: str,
                     val_annotations_file: str,
                     train_batch_size: int,
                     train_num_workers: int,
                     val_batch_size: int,
                     val_num_workers: int,
                     **hyper_params
                     ) -> Tuple[torch.utils.data.DataLoader,
                                torch.utils.data.DataLoader]:
    """ Create dataloaders for training and validation.

    Args:
        root: path to the root folder of the images and annotations.
        train_annotations_file: path to the train annotations from root
        val_annotations_file: path to the validation annotations from root
        train_batch_size: size of the batches used during training.
        train_num_workers: number of workers used during training.
        val_batch_size: size of the batches used during validation.
        val_num_workers: number of workers used during validation.

    Returns:
        tuple of first the dataloader for training and second the dataloader for
            validation.
    """

    dataset = GreatBarrierReefDataset(root=root,
                                      annotation_path=train_annotations_file,
                                      transforms=get_transform(True, **hyper_params))

    data_loader_train = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=train_num_workers, collate_fn=collate_fn, pin_memory=True)

    dataset_val = GreatBarrierReefDataset(root=root,
                                          annotation_path=val_annotations_file,
                                          transforms=get_transform(False, **hyper_params))
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=val_batch_size, shuffle=False,
        num_workers=val_num_workers, collate_fn=collate_fn, pin_memory=True)

    return data_loader_train, data_loader_val


def save_model(epoch: int, model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
               checkpoint_path: str,
               name: str) -> None:
    """ Saves the model in the checkpoint_path as Epoch_*.pth.

    Args:
        epoch: epoch
        model: Pytorch model
        optimizer: Pytorch optimizer
        lr_scheduler: Pytorch learning rate scheduler
        checkpoint_path: path to the folder where the checkpoint should be saved to
        name: file name to which the model should be saved to
    """

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict()
    }, os.path.join(checkpoint_path, name))

    # artifact = wandb.Artifact(name, type="model")
    # artifact.add_file(os.path.join(checkpoint_path, name), name=name)
    # wandb.log_artifact(artifact)


def print_loss(loss_dict: Dict[str, float], epoch: int, step: int,
               total_steps: int) -> None:
    """ Prints the formatted loss.

    Args:
        loss_dict: contains names and values of important losses.
        epoch: number of epoch
        step: number of step
        total_steps: number of total steps trained in one epoch
    """
    string_length = str(len(str(total_steps)))
    out_string = ('Epoch {epoch}: [{step:' + string_length + 'd}/{total_steps}]') \
        .format(epoch=epoch,
                step=step,
                total_steps=total_steps)
    out_string += ''.join(f' | {key}: {value:.5f}' for key, value in loss_dict.items())
    print(out_string)


def train_one_step(model: torch.nn.Module, images: torch.FloatTensor,
                   targets: List[Dict[str, torch.Tensor]],
                   optimizer: torch.optim.Optimizer,
                   gradient_clipping_norm: Optional[float]) -> Dict[str, float]:
    """ Trains the model for one step.

    Args:
        model: PyTorch model that returns the loss_dict in training mode
            (see e.g. https://pytorch.org/vision/main/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)
        images: (shape [B, C, H, W]) images to be processed, should be in 0-1 range
        targets: ground-truth annotations present in the images, should contain
            annotations (FloatTensor[N, 4]): the ground-truth annotations in
                [xmin, ymin, xmax, ymax] format, with values
                0 <= xmin < xmax <= W, 0 <= ymin < ymax <= H
            image_id (Int64Tensor[1]): image id
        optimizer: PyTorch optimizer
        gradient_clipping_norm: (optional) threshold to which the gradients are
            clipped if their norm is larger than this

    Returns:
        loss values from the model
    """
    loss_dict = model(images, targets)
    if 'total_loss' not in loss_dict.keys():
        loss_dict['total_loss'] = sum(loss for loss in loss_dict.values())

    loss = loss_dict['total_loss']

    loss_value = loss.item()

    if not math.isfinite(loss_value):
        print("Loss is {}, stopping training".format(loss_value))
        sys.exit(1)
    optimizer.zero_grad()
    loss.backward()
    if gradient_clipping_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_norm)
    optimizer.step()
    loss_dict_items = {k: v.item() for k, v in loss_dict.items()}
    return loss_dict_items


def train_and_evaluate(model: torch.nn.Module,
                       root: str,
                       train_annotations_file: str,
                       val_annotations_file: str,
                       num_epochs: int,
                       checkpoint_path: str,
                       skip_initial_val: bool = False,
                       train_batch_size: int = 32,
                       train_num_workers: int = 4,
                       val_batch_size: int = 4,
                       val_num_workers: int = 4,
                       gradient_clipping_norm: Optional[float] = None,
                       learning_rate: float = 0.0005,
                       eval_every_n_epochs: int = 1,
                       save_every_n_epochs: int = 1,
                       steps_without_improvement: int = 20,
                       keep_last_n_checkpoints: int = 10,
                       existing_checkpoint_path: str = None,
                       **hyper_params):
    """ Trains and evaluates the model.

    Trains for num_epoch epochs, evaluates every eval_every_n_epochs, saves the
    model every save_every_n_epochs and deletes the older checkpoints such that
    at maximum keep_last_n_checkpoints checkpoints exist in parallel.
    Clips the gradient at gradient_clipping_norm if it is set.
    Loads the model from existing_checkpoint_path if it is set and it creates a
    new folder for the checkpoints in checkpoint_path.

    Args:
        model: PyTorch model
        root: path to the root folder of the images and annotations
        train_annotations_file: path to the annotation file used for training (from root)
        val_annotations_file: path to the annotation file used for validations (from root)
        num_epochs: number of epochs for which the model should be trained
        checkpoint_path: path where the checkpoints and tensorboard files should
            be saved to
        train_batch_size: size of the batches used during training.
        train_num_workers: number of workers used during training.
        val_batch_size: size of the batches used during validation.
        val_num_workers: number of workers used during validation.
        gradient_clipping_norm: threshold to which the gradients are clipped if
            their norm is larger than this (optional)
        learning_rate: learning rate for the optimizer
        skip_initial_val: whether to skip the initial validation step
        eval_every_n_epochs: every eval_every_n_epochs epochs the model is evaluated
        save_every_n_epochs: every save_every_n_epochs epochs the model is saved
        steps_without_improvement: number of evaluations that are done without
            improving the metric_for_best_model before early stopping
        keep_last_n_checkpoints: keep_last_n_checkpoints checkpoints are left,
            all other are deleted
        existing_checkpoint_path: path to existing checkpoint that is used for
            training (optional)
    """
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_loader_train, data_loader_val = get_data_loaders(root,
                                                          train_annotations_file,
                                                          val_annotations_file,
                                                          train_batch_size,
                                                          train_num_workers,
                                                          val_batch_size,
                                                          val_num_workers)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params,
                                 lr=learning_rate,
                                 betas=(hyper_params.get("beta_1", 0.9), hyper_params.get('beta_2', 0.999)),
                                 weight_decay=hyper_params.get("weight_decay", 0))
    total_steps = len(data_loader_train)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    if existing_checkpoint_path:
        print('Loading checkpoint from ', existing_checkpoint_path)
        checkpoint = torch.load(existing_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    metric_for_best_model = 'F2 score'
    metric_should_be_large = True
    early_stopping_coefficient = 1 if metric_should_be_large else -1
    best_value = -np.inf * early_stopping_coefficient
    last_update = 0

    wandb.run.summary["best_model_metric"] = metric_for_best_model

    """First validation run"""
    if not hyper_params.get("skip_initial_val", False):
        _, _ = evaluate_and_plot(model, data_loader_val,
                                                      device=device)

        wandb.log({
            "epoch": 0,
            "global_step": 0,
            "val_step": 0,
        })

    """END"""

    for epoch in range(start_epoch + 1, num_epochs + 1):
        global_step = (epoch - 1) * total_steps

        learning_rate = lr_scheduler.state_dict()['_last_lr'][0]

        wandb.log({
            "learning_rate": learning_rate,
            "epoch": epoch,
            "global_step": global_step
        })

        model.train()
        for step, (images, targets) in enumerate(data_loader_train):
            global_step = (epoch - 1) * total_steps + step + 1
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in
                       targets]
            loss_dict = train_one_step(model, images, targets, optimizer,
                                       gradient_clipping_norm)

            wandb.log({
                "epoch": epoch,
                "batch": step + 1,
                "global_step": global_step,
                **{
                    "training/" + key: value
                    for key, value in loss_dict.items()
                }
            })

            print_loss(loss_dict, epoch, step + 1, total_steps)

        lr_scheduler.step()

        if epoch % eval_every_n_epochs == 0:
            val_metrics, val_log_dict = evaluate_and_plot(model,
                                                          data_loader_val,
                                                          device=device,
                                                          **hyper_params)

            wandb.log({
                "epoch": epoch,
                "global_step": epoch * total_steps,
                "val_step": epoch // eval_every_n_epochs,
                **{
                    "validation/" + key: value
                    for key, value in val_metrics.items()
                },
                **val_log_dict  # images, videos and plots
            })

            last_update += 1
            if bool(val_metrics) and early_stopping_coefficient * val_metrics[metric_for_best_model] > \
                    early_stopping_coefficient * best_value:
                save_model(epoch, model, optimizer, lr_scheduler,
                           checkpoint_path, 'best_model.pth')
                best_value = val_metrics[metric_for_best_model]
                wandb.run.summary["best_value"] = best_value
                last_update = 0

        if epoch != 0 and epoch % save_every_n_epochs == 0:
            print('Saving model...')
            save_model(epoch, model, optimizer, lr_scheduler,
                       checkpoint_path, 'Epoch_' + str(epoch) + '.pth')

            delete_epoch = epoch - save_every_n_epochs * keep_last_n_checkpoints
            if delete_epoch >= 0:
                delete_path = os.path.join(checkpoint_path, 'Epoch_' + str(delete_epoch) + '.pth')
                if os.path.exists(delete_path):
                    os.remove(delete_path)
            print('Model saved.')

        if last_update >= steps_without_improvement:
            print('Training stopped.')
            print(f'Best {metric_for_best_model}: {best_value}.')
            break

    wandb.finish()
