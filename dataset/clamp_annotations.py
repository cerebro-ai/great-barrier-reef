""" Created by Erik Steiger at 21.12.2021.
Removing all annotations that stand out from the picture.
"""

import numpy as np
import pandas as pd
import sys
import argparse

parser = argparse.ArgumentParser(description="""
    Takes a csv file with the column 'annotations' and clampes the width and heigth that it stays inside the image boundaries. 
    The new annotations are written to the column 'new_annotations' and potentially overwritten.
    """)

parser.add_argument("file", type=str, help="The input file. Should be a valid csv file.")
parser.add_argument("--out", default=None, help="The name of the file the new dataframe will be written to. [Default: input file]")

NEW_COLUMN_NAME = "clamped_annotations"
MAX_X = 1280
MAX_Y = 720



def extract(obj, key):
    if isinstance(obj, dict):
        return obj[key]
    else:
        return obj


def build_annotation(row):
    if pd.isna(row["annotations"]):
        return np.nan

    return {"x": int(row["x"]), "y": int(row["y"]), "width": int(row["new_width"]), "height": int(row["new_height"])}


def remove_nan_lists(x):
    if pd.isna(x[0]):
        return []
    else:
        return x


def create_new_annotations(df):
    # read and explode
    df["annotations"] = df["annotations"].apply(eval)

    df = df.explode("annotations")
    
    # extract x, y, width, height into seperate columns
    df["x"] = df["annotations"].apply(extract, key="x")
    df["y"] = df["annotations"].apply(extract, key="y")
    df["width"] = df["annotations"].apply(extract, key="width")
    df["height"] = df["annotations"].apply(extract, key="height")

    df["x2"] = df["x"] + df["width"]
    df["y2"] = df["y"] + df["height"]

    # create the clamped new coordinates
    df["clamped_x2"] = np.minimum(df["x2"], MAX_X)
    df["clamped_y2"] = np.minimum(df["y2"], MAX_Y)
    
    # for statistics only
    df["new_area"] = (df.clamped_x2 - df.x) * (df.clamped_y2-df.y)
    df["old_area"] = (df.x2 - df.x) * (df.y2-df.y)
    df["area_share"] = df["new_area"] / df["old_area"]
    assert np.all((df["area_share"]>0.3) | df["area_share"].isna())

    df["new_height"] = df["clamped_y2"] - df["y"]
    df["new_width"] = df["clamped_x2"] - df["x"]
    assert np.all((df.new_height <= df.height) | df.height.isna())
    assert np.all((df.new_width <= df.width) | df.width.isna())


    df[NEW_COLUMN_NAME] = df.apply(build_annotation, axis=1)

    # restore the structure with a new column: new_annotations
    new_df = df[["video_id", "sequence", "video_frame", "sequence_frame", "image_id", "annotations", NEW_COLUMN_NAME]]

    new_df = new_df\
        .groupby("image_id")\
        .agg({"video_id": "first", "sequence": "first", "video_frame": "first", "sequence_frame": "first", "annotations": list, NEW_COLUMN_NAME: list})\
        .reset_index()[["video_id", "sequence", "video_frame", "sequence_frame", "image_id", "annotations", NEW_COLUMN_NAME]]


    new_df["annotations"] = new_df["annotations"].apply(remove_nan_lists)
    new_df[NEW_COLUMN_NAME] = new_df[NEW_COLUMN_NAME].apply(remove_nan_lists)

    return new_df.sort_values(["video_id", "video_frame"]).reset_index(drop=True)


p = parser.parse_args()

df = pd.read_csv(p.file)
new_df = create_new_annotations(df)
changed_lines = new_df[new_df.annotations != new_df[NEW_COLUMN_NAME]]
print(f"{p.file}: {len(changed_lines)} frames got clamped")
out_file = p.out if p.out is not None else p.file
new_df.to_csv(f"{out_file}", index=False)