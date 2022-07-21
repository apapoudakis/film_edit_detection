import os
import xml.etree.ElementTree
import wget

import utils.video
from utils import video, annotation_utils
import re
import random


dataset_path = "../../Data/TRECVID/"
sep_string = "._-o-_."


def download_video():
    """
    Download TRECVID IACC.3 videos
    """

    video_path = os.path.join(dataset_path, "Video")

    if not os.path.exists(video_path):
        os.mkdir(video_path)

    trecvid = xml.etree.ElementTree.parse(os.path.join(dataset_path, "iacc.3.collection.xml"))

    for video in trecvid.getroot():
        source = video.find("source").text
        fn = video.find("filename").text.split(sep_string)[1]
        print(fn)

        if os.path.exists(os.path.join(video_path, fn)):
            continue

        if not os.path.exists(os.path.join(video_path, fn)):
            try:
                wget.download(os.path.join(source, fn), out=os.path.join(video_path, fn))
            except:
                print("Error!!!")


def create_train_dataset(videos_path, annotations_path, output_path, num_frames):
    """
    Trecvid as dataset using the already annotated cuts
    """
    videos = [x for x in os.listdir(videos_path) if x.endswith(".mp4")]
    idx = 0
    for v in videos:
        print(v)
        frames = video.get_frames(os.path.join(videos_path, v), 64, 64)

        annotations_file = annotation_utils.trecvid(v, annotations_path)
        with open(os.path.join(annotations_path, annotations_file), 'r', errors='replace') as f:
            boundaries = f.readlines()
            boundaries = [re.sub('\s+', ' ', x.strip()) for x in boundaries]

        for b in range(3, len(boundaries)-1):
            begin1 = int(boundaries[b].split(" ")[0])
            end1 = int(boundaries[b].split(" ")[1])
            begin2 = int(boundaries[b + 1].split(" ")[0])
            end2 = int(boundaries[b + 1].split(" ")[1])

            if begin2 - end1 == 1:
                start_frame = random.randint(end1-15, begin2+1)
                end_frame = start_frame + num_frames
                segment = frames[start_frame:end_frame]

                utils.video.array_to_video(segment, 20, output_path + "Hard/" + str(idx) + ".mp4")

            else:
                print("hi")
                start_frame = random.randint(end1 - 15, begin2)
                end_frame = start_frame + num_frames
                segment = frames[start_frame:end_frame]

                utils.video.array_to_video(segment, 20, output_path + "Hard" + str(idx) + ".mp4")
            idx += 1


if __name__ == "__main__":
    # download_video()

    create_train_dataset("../../Data/TRECVID/Video/", "../../Data/TRECVID/msb/", "../../Data/TRECVID/RealData", 16)

