"""
Converts annotation files in the csv format to json files in coco format
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description="""
    Converts annotation files in the csv format to json files in COCO format.
""")

parser.add_argument("input_file", type=str, help="Input file (.csv)")


def coco(df: pd.DataFrame) -> dict:
    annotation_id = 0
    images = []
    annotations = []

    annotation_column = "clamped_annotations" if "clamped_annotations" in df.columns else "annotations"
    df[annotation_column] = df[annotation_column].apply(eval)

    categories = [{'id': 0, 'name': 'cots'}]

    for i, row in tqdm(df.iterrows(), total=len(df)):

        images.append({
            "id": i,
            "file_name": f"video_{row['video_id']}/{row['video_frame']}.jpg",
            "height": 720,
            "width": 1280,
        })
        for bbox in row[annotation_column]:
            annotations.append({
                "id": annotation_id,
                "image_id": i,
                "category_id": 0,
                "bbox": list(bbox.values()),
                "area": bbox['width'] * bbox['height'],
                "segmentation": [],
                "iscrowd": 0
            })
            annotation_id += 1

    json_file = {'categories': categories, 'images': images, 'annotations': annotations}
    return json_file


def transform_file(file):
    df = pd.read_csv(file)
    coco_json = coco(df)
    with file.parent.joinpath("coco_" + file.stem + ".json").open("w") as f:
        json.dump(coco_json, f, ensure_ascii=True, indent=4)


p = parser.parse_args()
file = Path(p.input_file)
if file.is_dir():
    for f in file.glob("*.csv"):
        transform_file(f)
else:
    if file.is_file():
        transform_file(file)
