import numpy as np
import torch

from gbr.evaluate import Evaluator


def test_get_metrics():
    predictions_batches = [
        torch.tensor([
            [0, 0, 1, 1],
            [.5, 2, 2.5, 4],
            [4.5, 2.5, 5, 3],
            [6, 2, 7, 3]
        ], dtype=torch.float),
        torch.tensor([
            [0.5, 0, 1.5, 1],
            [4, 0, 5, 1],
            [0, 2, 1, 3],
            [4, 2, 5, 3]
        ], dtype=torch.float),
    ]
    scores_batches = [
        torch.tensor([
            .7,
            .5,
            .6,
            .9,
        ], dtype=torch.float),
        torch.tensor([
            .8,
            .3,
            .4,
            .2,
        ], dtype=torch.float),
    ]

    annotations_batches = [
        torch.tensor([
            [0, 2, 1, 3],
            [4, 2, 5, 3],
        ], dtype=torch.float),
        torch.tensor([
            [0, 0, 1, 1],
            [1, 0, 2, 1],
            [4, 0, 4.5, .5],
        ], dtype=torch.float),
    ]

    gt_precision = torch.tensor([
        [0, 0, 0, 0, 0],
        [.5, .5, .5, .5, .5],
        [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3],
        [.5, .5, .5, .5, .25],
        [.6, .4, .4, .4, .2],
        [.5, 1 / 3, 1 / 3, 1 / 3, 1 / 6],
        [4 / 7, 3 / 7, 3 / 7, 3 / 7, 1 / 7],
        [.5, 3 / 8, 3 / 8, 3 / 8, 1 / 8],
    ], dtype=torch.float)

    gt_recall = torch.tensor([
        [0, 0, 0, 0, 0],
        [.2, .2, .2, .2, .2],
        [.2, .2, .2, .2, .2],
        [.4, .4, .4, .4, .2],
        [.6, .4, .4, .4, .2],
        [.6, .4, .4, .4, .2],
        [.8, .6, .6, .6, .2],
        [.8, .6, .6, .6, .2],
    ], dtype=torch.float)

    iou_thresholds = torch.linspace(0.1, 0.3, int(np.round((0.3 - 0.1) / .05)) + 1)

    evaluator = Evaluator(torch.device('cpu'), iou_thresholds)
    for predictions, scores, annotations in zip(predictions_batches,
                                                scores_batches,
                                                annotations_batches):
        evaluator.update(predictions, scores, annotations)

    metrics, (recall_average, precision_average) = evaluator.accumulate()

    assert (precision_average == gt_precision.mean(dim=1)).all()
    assert (recall_average == gt_recall.mean(dim=1)).all()
