# TensorFlow - Help Protect the Great Barrier Reef

Implements methods to detect the crown-of-thorns starfish in underwater image data corresponding to the corresponding
[Kaggle competition](https://www.kaggle.com/c/tensorflow-great-barrier-reef).

## Prerequisites
Tested with
- Python 3.7
- PyTorch 1.10.0
- Tensorboard 2.7.0
  and the packages in `requirements.txt`.

## Data structure:
- dataset
    - train.csv
    - val.csv
    - train_images
        - video_0
            - 0.jpg
            - 1.jpg
            ...
        - video_1
            - 0.jpg
            - 1.jpg
            ...
        - video_2
            - 0.jpg
            - 1.jpg
            ...

## File structure:
- data/**gbr_dataset.py**: PyTorch dataset used in the dataloader (returns an image and the corresponding annotations)
- data/**transforms.py**: data augmentations
- **main.py**: main file to start the training
- **train.py**: training (and evaluation) loop
- **evaluation.py**: evaluation functions
- **tensorboard_utils.py**: functions to make plotting easier in Tensorboard
- **coco_eval.py**, **coco_utils.py**, **pytorch_utils.py**: Help methods from [Torchvision](https://github.com/pytorch/vision/tree/main/references/detection) 
  to evaluate object detectors with pycocotools