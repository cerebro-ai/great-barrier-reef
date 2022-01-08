from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.efficientnet import _efficientnet_conf

return_indices = [1, 2, 3, 5, 7, 8]
efficientnet_width_depth_dict = {
    'efficientnet_b0': dict(width_mult=1.0, depth_mult=1.0),
    'efficientnet_b1': dict(width_mult=1.0, depth_mult=1.1),
    'efficientnet_b2': dict(width_mult=1.1, depth_mult=1.2),
    'efficientnet_b3': dict(width_mult=1.2, depth_mult=1.4),
    'efficientnet_b4': dict(width_mult=1.4, depth_mult=1.8),
    'efficientnet_b5': dict(width_mult=1.6, depth_mult=2.2),
    'efficientnet_b6': dict(width_mult=1.8, depth_mult=2.6),
    'efficientnet_b7': dict(width_mult=2.0, depth_mult=3.1),
}


def efficientnet_fpn(name, out_channels=256, **kwargs):
    return_layers = {str(i): str(i) for i in return_indices}
    inverted_residual_setting = _efficientnet_conf(**efficientnet_width_depth_dict[name])
    in_channels_list = [config.out_channels for i, config in enumerate(inverted_residual_setting) if
                        i + 1 in return_indices] + [inverted_residual_setting[-1].out_channels * 4]
    return BackboneWithFPN(eval(name)(**kwargs).features, return_layers, in_channels_list, out_channels)


if __name__ == '__main__':
    import torch
    for name in efficientnet_width_depth_dict.keys():
        fpn = efficientnet_fpn(name, out_channels=256)
        print({key: value.shape for key, value in fpn(torch.rand(1, 3, 224, 224)).items()})
