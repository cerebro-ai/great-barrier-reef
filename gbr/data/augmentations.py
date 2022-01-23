import math
from pathlib import Path

import numpy as np
import torch
import numpy
import matplotlib.pyplot as plt
from scipy import ndimage
from torchvision.utils import draw_bounding_boxes


def copy_paste_augmentation(image1: torch.Tensor, bboxes1: torch.Tensor, image2: torch.Tensor, bboxes2: torch.Tensor):
    """

    Args:
        image1: torch (3, h, w)
        bboxes1: (N, 4) with x y x2 y2
        image2: torch (3, h, w)
        bboxes2: (N, 4) with x y x2 y2

    Returns:
        image, bboxes

    """
    stride = 80

    image = image1.clone()
    bboxes = [box.tolist() for box in bboxes1]

    # create initial mask
    mask = numpy.zeros(image.shape[1:])
    for box in bboxes1:
        # mark all existing boxes
        x, y, x2, y2 = box.int().tolist()
        mask[y:y2, x:x2] = 1

    # iterate over donor bboxes
    for box in bboxes2:
        x, y, x2, y2 = box.int().tolist()
        max_h, max_w = image2.shape[1:]
        h, w = int(y2 - y), int(x2 - x)

        # we choose a larger h, w such that the gcd is not 1, and we have smaller kernel which is faster
        optimal_h, optimal_w = int(h + (stride - h % stride)), int(w + (stride - w % stride))

        h_diff, w_diff = optimal_h - h, optimal_w - w

        # sample margins

        snippet_x = np.random.randint(min(max(0, x2 - optimal_w), max_w-1), max(min(x, max_w - optimal_w), 0)+1)
        snippet_y = np.random.randint(min(max(0, y2 - optimal_h), max_h-1), max(min(y, max_h - optimal_h), 0)+1)

        ml = x - snippet_x
        mt = y - snippet_y

        mb = h_diff - mt
        mr = w_diff - ml

        assert ml + mr == w_diff
        assert mb + mt == h_diff

        gcd = math.gcd(optimal_h, optimal_w)
        kernel_h, kernel_w = int(optimal_h / gcd), int(optimal_w / gcd)
        kernel = np.ones((2 * kernel_h + 1, 2 * kernel_w + 1)).astype(bool)
        mask_dil = ndimage.binary_dilation(mask,
                                           structure=kernel,
                                           iterations=int(optimal_w / kernel_w / 2),
                                           border_value=1).astype(mask.dtype)

        # sample random point where mask == 0
        if np.any(mask_dil == 0):
            idxs = numpy.where(mask_dil == 0)
            n = len(idxs[0])
            i = numpy.random.randint(0, n)
            cy, cx = idxs[0][i], idxs[1][i]

            # place the snipped center onto (cx, cy)

            # place the box onto the image
            a, b = int(cy - optimal_h / 2), int(cy + optimal_h / 2)
            c, d = int(cx - optimal_w / 2), int(cx + optimal_w / 2)
            image[:, a:b, c:d] = image2[:, y - mt:y2 + mb, x - ml:x2 + mr]

            # update the bboxes
            bboxes.append([c + ml, a + mt, d - mr, b - mb])

            # update the mask
            mask[a:b, c:d] = 1

        else:
            # there is no space for this box, skip it
            # TODO potentially make the box smaller and try again?
            continue
    return image, torch.tensor(bboxes).view(-1, 4)


def draw_image(image):
    plt.imshow(image)
    plt.show()


def draw_torch_image(image, bboxes=None):
    """

    Args:
        image: torch image 0,1 normalized with (3,
        bboxes:

    Returns:

    """
    image = (image * 255).to(device=torch.device('cpu'), dtype=torch.uint8)
    if bboxes is not None:
        image = draw_bounding_boxes(image, bboxes, width=2, colors="red")
    draw_image(torch.permute(image, (1, 2, 0)).numpy())


def dev_copy_paste():
    """
        image: torch.tensor with (C, Height, Width) normalized between 0,1
        bboxes: torch.tensor (N, 4) with [x, y, x2, y2] in pixel space
        """
    mask = numpy.zeros((512, 512))

    # x y x2 y2
    test_box = torch.tensor([100, 100, 200, 200])
    x, y, x2, y2 = test_box.tolist()

    # mark the box
    mask[y:y2, x:x2] = 1

    # box to insert
    h, w = 300, 100
    gcd = math.gcd(h, w)
    hi, hw = int(h / gcd), int(w / gcd)

    # expand the box with margin m
    # struct = ndimage.generate_binary_structure(2, 2)
    struct = numpy.ones((2 * hi + 1, 2 * hw + 1)).astype(bool)
    mask_dil = ndimage.binary_dilation(mask, structure=struct, iterations=int(w / hw / 2), border_value=1).astype(
        "float")

    # sample a random point
    idxs = numpy.where(mask_dil == 0)
    n = len(idxs[0])
    i = numpy.random.randint(0, n)

    cy, cx = idxs[0][i], idxs[1][i]
    print(cy, cx)

    pr = 100
    mask_dil[cy - int(h / 2):cy + int(h / 2), cx - int(w / 2):cx + int(w / 2)] = 0.5

    draw_image(mask_dil)


if __name__ == '__main__':
    from gbr.data.gbr_dataset import GreatBarrierReefDataset, get_transform

    path = Path(__file__).parent.parent.parent.joinpath("dataset").absolute()
    gbr_dataset = GreatBarrierReefDataset(root=str(path),
                                          annotation_path=str(Path(path).joinpath("reef_starter_0.05/train_clean.csv")),
                                          transforms=get_transform(True),
                                          copy_paste=False,
                                          apply_mixup=False)

    n = np.random.randint(0, len(gbr_dataset))
    m = np.random.randint(0, len(gbr_dataset))
    image1, target1 = gbr_dataset[n]
    image2, target2 = gbr_dataset[m]
    # image3, target3 = gbr_dataset[1700]

    # draw_torch_image(image1, target1["boxes"])
    # draw_torch_image(image2, target2["boxes"])

    image, target = copy_paste(image1, target1["boxes"], image2, target2["boxes"])
    # image, target = copy_paste(image, target, image3, target3["boxes"])

    draw_torch_image(image, target)

    # draw_image(image2)
