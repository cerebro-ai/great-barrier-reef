from typing import Dict

import numpy as np
import torch

import wandb


class MultipleVideoBuffer:
    """Wrapper around a dictionary of video buffers"""

    def __init__(self, max_frames=600, fps=18):
        self.max_frames = max_frames
        self.fps = fps
        self.video_buffers: Dict[str, VideoBuffer] = {}

    def append(self, key: str, value: np.ndarray):
        try:
            if key not in self.video_buffers.keys():
                self.video_buffers[key] = VideoBuffer(max_frames=self.max_frames, fps=self.fps)
            self.video_buffers[key].append(value)
        except Exception:
            pass

    def reset(self):
        for buffer in self.video_buffers.values():
            buffer.reset()

    def export(self):
        videos = {}
        try:
            for key, buffer in self.video_buffers.items():
                videos[key] = buffer.export()
            return videos
        except:
            return {}


class VideoBuffer:
    """Buffers images as numpy arrays and exports them as videos

    """

    def __init__(self, max_frames=600, fps=10):
        """
        Args:
            max_frames: Max number of frames per video, if more frames are passed, a new video is started
            fps: Frames per second of the generated video
        """
        self.max_frames = max_frames
        self.fps = fps
        self.frames = []  # holds the frames of the current video
        self.videos = []  # holds the video objects

    def append(self, image: np.ndarray):
        """Add a new frame at the end of the current frame buffer"""
        self.frames.append(image)
        if len(self.frames) >= self.max_frames:
            # generate video
            video = wandb.Video(np.stack(self.frames), fps=self.fps, format="mp4")
            self.videos.append(video)
            # reset frames buffer
            self.frames = []

    def reset(self):
        """Resets all buffers"""
        self.frames = []
        self.videos = []

    def export(self):
        """Generates the last video from the frames buffer and returns all videos"""
        if len(self.frames) > 0:
            video = wandb.Video(np.stack(self.frames), fps=self.fps, format="mp4")
            self.videos.append(video)
            # reset frames buffer
            self.frames = []
        return self.videos


def create_box_data(box_data, mode: str = "prediction"):
    """Creates the box_data given target or output

    Args:
        box_data: target or output with key 'boxes'
        mode:

    Returns:

    """
    assert mode in ["prediction", "ground_truth"]
    is_pred = True if mode == "prediction" else False
    boxes = []
    for i, box in enumerate(box_data["boxes"]):
        img_box = box.to(torch.int64).tolist()
        if is_pred:
            score = round(box_data["scores"][i].item(), 2)
        else:
            score = 1

        wandb_box_dict = {
            "position": {"minX": img_box[0], "minY": img_box[1], "maxX": img_box[2], "maxY": img_box[3]},
            "class_id": 2 if is_pred else 1,
            "box_caption": str(score) if is_pred else "starfish",
            "scores": {"score": score},
            "domain": "pixel"
        }
        boxes.append(wandb_box_dict)
    return boxes
