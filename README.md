# TensorFlow - Help Protect the Great Barrier Reef

Implements methods to detect the crown-of-thorns starfish in underwater image data corresponding to the corresponding
[Kaggle competition](https://www.kaggle.com/c/tensorflow-great-barrier-reef).

## Prerequisites

Tested with

- Python 3.7
- PyTorch 1.10.0
- Tensorboard 2.7.0 and the packages in `requirements.txt`.

## Data structure:

- dataset
    - train.csv
    - val.csv
    - train_images
        - video_0
            - 0.jpg
            - 1.jpg ...
        - video_1
            - 0.jpg
            - 1.jpg ...
        - video_2
            - 0.jpg
            - 1.jpg ...

## File structure:

- data/**gbr_dataset.py**: PyTorch dataset used in the dataloader (returns an image and the corresponding annotations)
- data/**transforms.py**: data augmentations
- **main.py**: main file to start the training
- **train.py**: training (and evaluation) loop
- **evaluation.py**: evaluation functions
- **tensorboard_utils.py**: functions to make plotting easier in Tensorboard
- **coco_eval.py**, **coco_utils.py**, **pytorch_utils.py**: Help methods
  from [Torchvision](https://github.com/pytorch/vision/tree/main/references/detection)
  to evaluate object detectors with pycocotools

## Models

Currently, supported models include:

- FasterRCNN with backbone
    - Resnet (resnet18, resnet50)
    - EfficientNet (efficientnet-d0, efficientnet-d4)
- YOLOX with CSP-Backbone (yolox-s, yolox-m)

## Config

Create the config file with the name `config.yaml` in the same directory as `main.py`. The variables under **params**
will be logged.

```yaml
# config.yaml

local:
  dataset_root: "/home/.../great-barrier-reef/dataset"
  train_annotations: "/home/.../dataset/reef_starter_0.05/train.csv"
  val_annotations: "/home/.../dataset/reef_starter_0.05/val.csv"

  checkpoint_root: "/home/.../great-barrier-reef/checkpoints"
  resume_checkpoint: "2021-12-17T23_32_18/best_model.pth"
  pretrained_weights_root: "/home/.../pretrained_weights"

params:
  model_name: yolox-m
  num_epochs: 40
  eval_every_n_epochs: 1
  save_every_n_epochs: 10
  train_batch_size: 2
  val_batch_size: 1
  train_num_workers: 0
  val_num_workers: 0

  optimizer: "Adam" # SGD / Adam
  learning_rate: 1.e-4
  weight_decay: 0.0005
  momentum: 0.9
  dampening: 0
  nesterov: False
  beta_1: 0.9 # Adam
  beta_2: 0.999 # Adam

  gradient_clipping_norm: 35

  input_size: [ 512, 512 ]  # height, width
  test_size: [ 736, 1312 ]  # height, width
  nms_thresh: 0.65

  rotation_limit: 10
  random_scale: 0.2
  random_rain_prob: 0.2
  use_copy_paste: True
```