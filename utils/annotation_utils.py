import os


def trecvid(video_name, annotations_path):
    """
    Given a video name returns the corresponding annotation file
    """
    files = os.listdir(annotations_path)
    for f in files:
        if f.startswith(video_name.split("_512kb")[0]) or f.endswith(video_name.split(".mp4")[0] + ".msb") \
                or f.endswith(video_name + "_512kb.msb"):
            return f

    return 1
