from torchvision.models import resnet
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign

from backbones.efficientnet import efficientnet_fpn, efficientnet_width_depth_dict

fasterrcnn_params = dict(
    image_mean=[0.2652, 0.5724, 0.6195],
    image_std=[0.2155, 0.1917, 0.1979],
    box_nms_thresh=.35,
    box_score_thresh=.0001,
    min_size_train=512, max_size_train=512,
    min_size_test=720, max_size_test=1280,
)


def fasterrcnn_fpn(name, **kwargs):
    """

    Args:
        name: one of 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
            'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
            'efficientnet_b6', 'efficientnet_b7', 'resnet18', 'resnet34',
            'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d',
            'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
    """
    if name in efficientnet_width_depth_dict.keys():
        return fasterrcnn_efficientnet_fpn(name, **kwargs)
    elif name in resnet.__all__:
        return fasterrcnn_resnet_fpn(name, **kwargs)
    else:
        raise ValueError(f'{name} not known')


def fasterrcnn_efficientnet_fpn(name,
                                featmap_names=('1', '2', '3', '5', '7', '8'),
                                sizes=((8, 16, 32), (16, 32, 64), (16, 32, 64), (16, 32, 64),
                                       (32, 64, 128), (32, 64, 128), (64, 128, 256)),
                                aspect_ratios=((.5, 1.0, 2.),) * 7,
                                **fasterrcnn_kwargs):
    """

    Args:
        name: one of 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
            'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
            'efficientnet_b6', 'efficientnet_b7'
    """
    backbone = efficientnet_fpn(name)
    return fasterrcnn(backbone, featmap_names, sizes, aspect_ratios, **fasterrcnn_kwargs)


def fasterrcnn_resnet_fpn(name,
                          trainable_layers=5,
                          featmap_names=('0', '1', '2', '3'),
                          sizes=((8, 16, 32), (16, 32, 64), (16, 32, 64), (32, 64, 128), (64, 128, 256)),
                          aspect_ratios=((.5, 1.0, 2.),) * 5,
                          **fasterrcnn_kwargs):
    """

    Args:
        name: one of 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2'
    """
    backbone = resnet_fpn_backbone(name, True, trainable_layers=trainable_layers)
    return fasterrcnn(backbone, featmap_names, sizes, aspect_ratios, **fasterrcnn_kwargs)


def fasterrcnn(backbone, featmap_names, sizes, aspect_ratios, **fasterrcnn_kwargs):
    # let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios
    anchor_generator = AnchorGenerator(sizes=sizes,
                                       aspect_ratios=aspect_ratios)

    # let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # if your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = MultiScaleRoIAlign(featmap_names=featmap_names,
                                    output_size=7,
                                    sampling_ratio=2)

    kwargs = dict(fasterrcnn_params, **fasterrcnn_kwargs)
    # put the pieces together inside a FasterRCNN model
    model = FasterRCNNBetter(backbone,
                             num_classes=2,
                             rpn_anchor_generator=anchor_generator,
                             box_roi_pool=roi_pooler,
                             **kwargs)

    return model


class FasterRCNNBetter(FasterRCNN):
    def __init__(self, *args,
                 min_size_train=512, max_size_train=512,
                 min_size_test=720, max_size_test=1280,
                 **kwargs):
        self.min_size_test = min_size_test
        self.max_size_test = max_size_test

        self.min_size_train = kwargs['min_size'] = min_size_train
        self.max_size_train = kwargs['max_size'] = max_size_train
        super().__init__(*args, **kwargs)

    def train(self, mode: bool = True):
        self.transform.min_size = (self.min_size_train,) if mode else (self.min_size_test,)
        self.transform.max_size = self.max_size_train if mode else self.max_size_test
        super().train(mode)
