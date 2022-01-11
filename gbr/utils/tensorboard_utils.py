""" Created by Dominik Schnaus at 13.12.2021.
Utils for visualizing everything easier with tensorboard
"""
from typing import Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def plot_scalars_in_tensorboard(header: str, metrics: Dict[str, float],
                                writer: torch.utils.tensorboard.SummaryWriter,
                                global_step: int) -> None:
    """ Plots scalar values metrics in tensorboard with additional header.

    Args:
        header: string that is shown in front of each key
        metrics: name -> value
        writer: tensorboard summary writer
        global_step: step to show in tensorboard
    """
    for key, value in metrics.items():
        writer.add_scalar(header + key, value, global_step)


def plot_images_in_tensorboard(results: Dict[int, np.ndarray],
                               writer: torch.utils.tensorboard.SummaryWriter,
                               global_step: int) -> None:
    """ Plots images in tensorboard.
    Args:
        results: image id -> image with predictions
        writer: tensorboard summary writer
        global_step: step to show in tensorboard
    """
    for key, value in results.items():
        writer.add_image(str(key), value, global_step=global_step,
                         dataformats='CHW')


def plot_pr_curve_in_tensorboard(pr_curve_dict: Dict[str, np.ndarray],
                                 writer: torch.utils.tensorboard.SummaryWriter,
                                 global_step: int) -> None:
    """ Plots a Precision-Recall-Curve in tensorboard.
    Args:
        pr_curve_dict: dict containing 'True Positives', 'False Positives',
            'True Negatives', 'True Negatives', 'False Negatives',
            'Precision', 'Recall'
        writer: tensorboard summary writer
        global_step: step to show in tensorboard
    """
    writer.add_pr_curve_raw('PR-Curve',
                            true_positive_counts=pr_curve_dict['True Positives'],
                            false_positive_counts=pr_curve_dict['False Positives'],
                            true_negative_counts=np.zeros_like(pr_curve_dict['Precision']),
                            false_negative_counts=pr_curve_dict['False Negatives'],
                            precision=pr_curve_dict['Precision'],
                            recall=pr_curve_dict['Recall'],
                            global_step=global_step,
                            num_thresholds=pr_curve_dict['Precision'].shape[0])


def plot_histogram_in_tensorboard(threshold_values: np.ndarray,
                                  writer: torch.utils.tensorboard.SummaryWriter,
                                  global_step: int) -> None:
    """ Plots a histogram in tensorboard as image (exact) and as histogram
    (compressed and inexact).

    Args:
        threshold_values: array containing all threshold values
        writer: tensorboard summary writer
        global_step: step to show in tensorboard
    """
    f = plt.figure()
    values, counts = np.unique(threshold_values, return_counts=True)
    plt.vlines(values, 0, counts,
               color=(238 / 255, 119 / 255, 51 / 255, 1.))  # tensorboard orange
    plt.xticks([0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
    writer.add_figure('optimal_threshold_histogram', f, global_step)
    writer.add_histogram('optimal_threshold', threshold_values, 0,
                         bins=np.arange(0.0, 1.01, 0.01))
