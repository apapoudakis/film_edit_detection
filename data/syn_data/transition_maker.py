import os
import random
import pandas as pd
from utils import video, annotation_utils
import re
from data.syn_data.synthetic_data import SyntheticVideoData


def select_random_shot(frames, video_gt, num_frames):
    """
    Select randomly a shot from a video

    :param frames:
    :param video_gt:
    :param num_frames:
    :return:
    """
    while 1:
        idx1 = random.randrange(3, len(video_gt))
        begin1 = int(video_gt[idx1].split(" ")[0])
        end1 = int(video_gt[idx1].split(" ")[1])
        start_frame = random.randint(begin1, end1)
        if (start_frame + num_frames) < end1 and frames[start_frame:start_frame + num_frames, :, :, :].shape[0] == num_frames:
            shot = frames[start_frame:start_frame + num_frames, :, :, :]
            break
    return shot


def select_random_video(video_files, annotation_per_film_path):
    while 1:
        video = random.choice(video_files)
        video_name = video.split("_512kb")[0]
        with open(
                os.path.join(annotation_per_film_path, annotation_utils.trecvid(video_name, annotation_per_film_path)),
                'r', errors='replace') as f:
            video_gt = f.readlines()
            video_gt = [re.sub('\s+', ' ', x.strip()) for x in video_gt[5:]]

        if len(video_gt) >= 10: break
    return video, video_name, video_gt


def generate_random_shots(videos_path, annotation_per_film_path, num_frames, num_cuts, output_path, idx, type_of_cut):
    """
    Select two random shots ensuring that they do not contain cut

    :param type_of_cut:
    :param idx:
    :param output_path:
    :param videos_path:
    :param num_cuts:
    :param annotation_per_film_path:
    :param num_frames:
    :return:
    """

    video_files = [x for x in os.listdir(videos_path) if x.endswith(".mp4")]

    video1, video_name1, video1_gt = select_random_video(video_files, annotation_per_film_path)
    video2, video_name2, video2_gt = select_random_video(video_files, annotation_per_film_path)

    frames1 = video.get_frames(os.path.join(videos_path, video1), 64, 64)
    frames2 = video.get_frames(os.path.join(videos_path, video2), 64, 64)

    df = pd.DataFrame(columns=["Film1", "Film2", "Type of Cut"])
    if not os.path.exists(os.path.join(output_path, "annotations.csv")):
        df.to_csv(os.path.join(output_path, "annotations.csv"), index=False)

    for i in range(num_cuts):

        # Shot1
        shot1 = select_random_shot(frames1, video1_gt, num_frames)
        shot2 = select_random_shot(frames2, video2_gt, num_frames)

        new_row = [video_name1, video_name2, type_of_cut]
        df.loc[len(df)] = new_row

        if random.choice([True, False]):
            shot1, shot2 = shot2, shot1

        syn = SyntheticVideoData(shot1, shot2, 16)
        frames = syn.generate_transition(type_of_cut)
        # video.array_to_video(frames, 10, os.path.join(output_path, type_of_cut, str(idx) + ".mp4"))
        idx = idx + 1

    # df.to_csv(os.path.join(output_path, "annotations.csv"), mode="a", index=False, header=False)


if __name__ == "__main__":

    type_of_transitions = ["Gradual", "Hard", "No Transition"]
    samples_counter = 0
    for t in type_of_transitions:
        print(t)
        for i in range(2000):
            print(i)
            generate_random_shots(videos_path="../../../Data/TRECVID/Video",
                                  annotation_per_film_path="../../../Data/TRECVID/msb",
                                  num_frames=16, num_cuts=50, output_path="../../../Data/TRECVID/SyntheticDataset/",
                                  idx=samples_counter,
                                  type_of_cut=t)
            samples_counter += 50
