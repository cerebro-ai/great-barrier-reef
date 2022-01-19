# -*- coding: utf-8 -*-
# @Time    : 2021/7/21 22:00
# @Author  : MingZhang
# @Email   : zm19921120@126.com

import time

import cv2
import numpy as np
import torch
import torch.nn as nn

from gbr.backbones.csp_darknet import CSPDarknet
from gbr.models.neck.yolo_fpn import YOLOXPAFPN
from gbr.models.head.yolo_head import YOLOXHead
from gbr.models.losses import YOLOXLoss
from gbr.models.post_process import yolox_post_process
from gbr.models.ops import fuse_model
# from data.data_augment import preproc
# from utils.model_utils import load_model
from gbr.utils.util import sync_time


def get_model(opt):
    # define backbone
    backbone_cfg = {"nano": [0.33, 0.25],
                    "tiny": [0.33, 0.375],
                    "s": [0.33, 0.5],
                    "m": [0.67, 0.75],
                    "l": [1.0, 1.0],
                    "x": [1.33, 1.25]}
    depth, width = backbone_cfg[opt.backbone.split("-")[1]]  # "CSPDarknet-s
    in_channel = [256, 512, 1024]
    backbone = CSPDarknet(dep_mul=depth, wid_mul=width, out_indices=(3, 4, 5), depthwise=opt.depth_wise)
    # define neck
    neck = YOLOXPAFPN(depth=depth, width=width, in_channels=in_channel, depthwise=opt.depth_wise)
    # define head
    head = YOLOXHead(num_classes=opt.num_classes, reid_dim=opt.reid_dim, width=width, in_channels=in_channel,
                     depthwise=opt.depth_wise)
    # define loss
    loss = YOLOXLoss(opt.label_name, reid_dim=opt.reid_dim, id_nums=opt.tracking_id_nums, strides=opt.stride,
                     in_channels=in_channel)
    # define network
    model = YOLOX(opt, backbone=backbone, neck=neck, head=head, loss=loss)
    return model


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


class YOLOX(nn.Module):
    def __init__(self, opt, backbone, neck, head, loss):
        super(YOLOX, self).__init__()
        self.opt = opt
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss = loss

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def normalize_batch(self, images):
        dtype, device = images.dtype, images.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (images - mean[None, :, None, None]) / std[None, :, None, None]

    def forward(self, inputs, targets: torch.Tensor = None, show_time=False):

        # convert targets into yolox targets
        # List of dictionary -> Tensor(B, L, A)
        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        targets = to_yolox_targets(targets).to(device)

        if isinstance(inputs, list):
            for i in range(len(inputs)):
                image = inputs[i]
                image = self.normalize(image)
                inputs[i] = image
            inputs = torch.stack(inputs)
        else:
            inputs = self.normalize_batch(inputs)

        with torch.cuda.amp.autocast(enabled=self.opt.use_amp):
            if show_time:
                s1 = sync_time(inputs)

            body_feats = self.backbone(inputs)
            neck_feats = self.neck(body_feats)
            yolo_outputs = self.head(neck_feats)
            # print('yolo_outputs:', [[i.shape, i.dtype, i.device] for i in yolo_outputs])  # float16 when use_amp=True

            if show_time:
                s2 = sync_time(inputs)
                print("[inference] batch={} time: {}s".format("x".join([str(i) for i in inputs.shape]), s2 - s1))

            if targets is not None:
                loss = self.loss(yolo_outputs, targets)
                # for k, v in loss.items():
                #     print(k, v, v.dtype, v.device)  # always float32

        if targets is not None:
            if self.training:
                return yolo_outputs, loss
            else:
                return yolox_post_process(yolo_outputs, self.opt.stride,
                                          self.opt.num_classes,
                                          conf_thre=self.opt.vis_thresh,
                                          nms_thre=self.opt.nms_thresh,
                                          label_name=self.opt.label_name,
                                          img_ratios=[1] * inputs.shape[0],
                                          img_shape=[img.shape[1:] for img in inputs]
                                          )
        else:
            return yolo_outputs


def to_yolox_targets(targets):
    MAX_LABELS = 10

    all_labels = []
    for target in targets:
        boxes = xyxy2cxcywh(target["boxes"])
        labels = torch.hstack([target["labels"].unsqueeze(1), boxes])
        padded_labels = torch.zeros((MAX_LABELS, 5))
        padded_labels[:len(target["labels"]), :] = labels
        all_labels.append(padded_labels)

    labels = torch.stack(all_labels)
    return labels


def xyxy2cxcywh(bboxes: torch.Tensor):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


class Detector(object):
    def __init__(self, cfg):
        self.opt = cfg
        self.opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.opt.pretrained = None
        self.model = get_model(self.opt)
        print("Loading model {}".format(self.opt.load_model))
        # self.model = load_model(self.model, self.opt.load_model)
        self.model.to(self.opt.device)
        self.model.eval()
        if "fuse" in self.opt and self.opt.fuse:
            print("==>> fuse model's conv and bn...")
            self.model = fuse_model(self.model)

    def run(self, images, vis_thresh, show_time=False):
        batch_img = True
        if np.ndim(images) == 3:
            images = [images]
            batch_img = False

        with torch.no_grad():
            if show_time:
                s1 = time.time()

            img_ratios, img_shape = [], []
            inp_imgs = np.zeros([len(images), 3, self.opt.test_size[0], self.opt.test_size[1]], dtype=np.float32)
            for b_i, image in enumerate(images):
                img_shape.append(image.shape[:2])
                # img, r = preproc(image, self.opt.test_size, self.opt.rgb_means, self.opt.std)
                img, r = None, None
                inp_imgs[b_i] = img
                img_ratios.append(r)

            if show_time:
                s2 = time.time()
                print("[pre_process] time {}".format(s2 - s1))

            inp_imgs = torch.from_numpy(inp_imgs).to(self.opt.device)
            yolo_outputs = self.model(inp_imgs, show_time=show_time)

            if show_time:
                s3 = sync_time(inp_imgs)
            predicts = yolox_post_process(yolo_outputs, self.opt.stride, self.opt.num_classes, vis_thresh,
                                          self.opt.nms_thresh, self.opt.label_name, img_ratios, img_shape)
            if show_time:
                s4 = sync_time(inp_imgs)
                print("[post_process] time {}".format(s4 - s3))
        if batch_img:
            return predicts
        else:
            return predicts[0]
