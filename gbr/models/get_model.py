from easydict import EasyDict

from gbr.models.faster_rcnn import fasterrcnn_fpn
from gbr.models.yolox import get_model as get_yolox_model
from gbr.utils.yolox_model_utils import load_model


def get_model(model_name: str, **config):
    """Given the model_name returns the corresponding model

    Args:
        model_name:
        **config: full config

    Returns:

    """
    if "yolo" in model_name.lower():

        model_size = model_name.split("-")[1]  # yolox-l
        opt = EasyDict(
            dict(
                backbone=f"CSPDarknet-{model_size}",
                depth_wise=True if model_size == "nano" else False,
                input_size=tuple(config["params"].get("input_size", (512, 512))),
                test_size=tuple(config["params"].get("input_size", (736, 1312))),
                num_classes=1,
                label_name=["cots"],
                reid_dim=0,
                stride=[8, 16, 32],
                vis_thresh=0.001,
                nms_thresh=config["params"].get("nms_thresh", 0.5),
                tracking_id_nums=None,
                use_amp=False
            )
        )
        yolo_model = get_yolox_model(opt)
        if config["local"].get("pretrained_weights_root", None):
            yolo_model = load_model(yolo_model, config["local"]["pretrained_weights_root"] + f"/{model_name}.pth")
        return yolo_model
    else:
        return fasterrcnn_fpn(model_name, **config["params"])
