""" Created by Erik Steiger at 15.12.2021.
Handling the upload of validation images
"""

# the image_ids as found in val.csv/image_id
image_ids_to_upload = ["0-4206", "0-4210"]


def img_id(image_id: str) -> int:
    """Transforms the image_id to an unique integer used inside batches as target["image_id"]

    "0-4206" ->    4206
    "1-2344" -> 1002344
    """
    video_id, frame_id = image_id.split("-")
    return int(f"{video_id}{int(frame_id):05d}")


