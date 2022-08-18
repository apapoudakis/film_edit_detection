import os
from evaluation.helpers import gt_boundaries
from utils.video import get_frames, get_fps, array_to_video, get_segment_frames
import torchaudio
import random
import pandas as pd


def select_random_shot(video_gt, num_frames, frames, audio_data, sr, fps):
    while 1:
        random_cut_idx = random.randrange(1, len(video_gt) - 1)
        next_cut_idx = random_cut_idx + 1

        _, end1, _ = video_gt[random_cut_idx]
        start2, _, _ = video_gt[next_cut_idx]

        if start2 - end1 >= num_frames:
            temp_start = random.randint(end1, start2 - num_frames)
            temp_end = temp_start + num_frames
            video_segment = frames[temp_start: temp_end, :, :, :]
            audio_segment = audio_data[:, temp_start * sr // fps: temp_end * sr // fps]
            break

    return video_segment, audio_segment


def data_formatting(root_dir, output_path, width, height, num_frames=16):
    video_files = [x for x in os.listdir(root_dir) if x.endswith(".mp4")]

    df = pd.DataFrame(columns=["Idx", "Type of Cut"])
    if not os.path.exists(os.path.join(output_path, "annotations.csv")):
        df.to_csv(os.path.join(output_path, "annotations.csv"), index=False)

    idx = 0

    for v in video_files:

        print(v)
        frames = get_frames(os.path.join(root_dir, v), width=width, height=height)
        audio, sr = torchaudio.load(os.path.join(root_dir, v.split(".mp4")[0] + ".wav"))

        fps = int(get_fps(os.path.join(root_dir, v)))
        gt_cuts = gt_boundaries(os.path.join(root_dir, v.split(".mp4")[0] + "_gt.txt"))
        for start_gt, end_gt, label in gt_cuts:
            if label == 1:
                start = random.randint(0, num_frames - (end_gt - start_gt + 1))
                end = num_frames - start
                video_segment = frames[start_gt - start: start_gt + end, :, :, :]
                audio_segment = audio[:, (start_gt - start) * sr // fps: (start_gt + end) * sr // fps]
                array_to_video(video_segment, 10, output_path + "/Hard/" + str(idx) + ".mp4")
                torchaudio.save(output_path + "/Hard/" + str(idx) + ".wav", audio_segment, sr)
                new_row = [str(idx), label]
                idx += 1
                df.loc[len(df)] = new_row
            else:

                if end_gt - start_gt < num_frames:
                    start = random.randint(0, num_frames - (end_gt - start_gt + 1))
                    end = num_frames - start
                    video_segment = frames[start_gt - start: start_gt + end, :, :, :]
                    audio_segment = audio[:, (start_gt - start) * sr // fps: (start_gt + end) * sr // fps]
                else:
                    start = random.randint(start_gt, end_gt - num_frames)
                    video_segment = frames[start: start + num_frames, :, :, :]
                    audio_segment = audio[:, start * sr // fps: (start + num_frames) * sr // fps]

                array_to_video(video_segment, 10, output_path + "/Gradual/" + str(idx) + ".mp4")
                torchaudio.save(output_path + "/Gradual/" + str(idx) + ".wav", audio_segment, sr)
                new_row = [str(idx), label]
                idx += 1
                df.loc[len(df)] = new_row

            # generate No Transitions Shots
            for i in range(10):
                video_segment, audio_segment = select_random_shot(gt_cuts, num_frames, frames, audio, sr, fps)
                array_to_video(video_segment, 10, output_path + "/No Transition/" + str(idx) + ".mp4")
                torchaudio.save(output_path + "/No Transition/" + str(idx) + ".wav", audio_segment, sr)
                idx += 1

    df.to_csv(os.path.join(root_dir, "annotations.csv"), mode="a", index=False, header=False)


data_formatting("../../Data/BBC_Planet_Earth_Dataset/video", "../../Data/BBC_Planet_Earth_Dataset/EditedBBC/", 112, 112)
