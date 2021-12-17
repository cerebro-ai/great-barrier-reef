import torch
from torchvision.ops import box_iou
import torch.utils.benchmark as benchmark

from evaluate import get_matched_indices


def generate_boxes(num_boxes,
                   height_fraction: float = 1 / 3,
                   width_fraction: float = 1 / 3,
                   height: int = 500,
                   width: int = 1000):
    boxes = torch.rand(num_boxes, 4) * \
            torch.tensor([[width, height,
                           width * width_fraction, height * height_fraction]])
    boxes[:, 2:] += boxes[:, :2]
    assert (boxes[:, 2:] > boxes[:, :2]).all()
    return boxes


def get_matched_indices_loop(t):
    matched_row_indices = []
    matched_column_indices = []
    columns = set()
    for row_index, row in enumerate(t):
        argmax = -1
        maximum = -1
        for column_index, element in enumerate(row):
            if column_index in columns or element == 0.:
                continue
            if element > maximum:
                maximum = element
                argmax = column_index
        if argmax != -1:
            matched_row_indices.append(row_index)
            matched_column_indices.append(argmax)
            columns.add(argmax)
        if len(columns) == t.shape[1]:
            break
    return torch.as_tensor(matched_row_indices, dtype=torch.long), \
           torch.as_tensor(matched_column_indices, dtype=torch.long)



if __name__ == '__main__':
    predictions = generate_boxes(1000, height_fraction=.5, width_fraction=.5)
    annotations = generate_boxes(1000, height_fraction=.5, width_fraction=.5)

    num_threads = torch.get_num_threads()
    device = torch.device('cuda')

    t = box_iou(predictions, annotations)
    t_copy = t.clone()

    t0 = benchmark.Timer(
        stmt='get_matched_indices(t)',
        setup='from __main__ import get_matched_indices',
        globals={'t': t_copy},
        num_threads=num_threads)

    t1 = benchmark.Timer(
        stmt='get_matched_indices_loop(t)',
        setup='from __main__ import get_matched_indices_loop',
        globals={'t': t},
        num_threads=num_threads)

    print(t0.blocked_autorange())
    print(t1.blocked_autorange())
