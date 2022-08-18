import os
import random
import pandas as pd
import numpy as np
from utils import annotation_utils, video
from scipy.io import wavfile
import re
import synthetic_data
import argparse


def select_random_shot(video_gt, num_frames, frames, video_path, audio, audio_data, sr):

    # offset between audio and visual signals
    offset = 100

    while 1:
        idx1 = random.randrange(3, len(video_gt))
        begin1 = int(video_gt[idx1].split(" ")[0])
        end1 = int(video_gt[idx1].split(" ")[1])
        start_frame = random.randint(begin1, end1)
        end_frame = start_frame + num_frames
        fps = int(video.get_fps(os.path.join(video_path, audio.split(".wav")[0] + ".mp4")))
        if (start_frame + num_frames) < end1 \
                and frames[start_frame:start_frame + num_frames, :, :, :].shape[0] == num_frames \
                and audio_data[(start_frame * sr // fps):(start_frame + num_frames) * sr // fps, :].shape[0] > 0:
            visual_shot = frames[start_frame:start_frame + num_frames, :, :, :]
            audio_shot = audio_data[((start_frame-offset) * sr // fps):(end_frame) * sr // fps, :]
            break
    return start_frame, end_frame, visual_shot, audio_shot


def select_random_video(video_files, annotation_per_film_path):
    while 1:
        video = random.choice(video_files)
        audio = video.split(".mp4")[0] + ".wav"
        audio_name = audio.split("_512kb")[0]
        video_name = audio_name

        with open(
                os.path.join(annotation_per_film_path, annotation_utils.trecvid(audio_name, annotation_per_film_path)),
                'r', errors='replace') as f:
            video_gt = f.readlines()
            video_gt = [re.sub('\s+', ' ', x.strip()) for x in video_gt[5:]]
        if len(video_gt) >= 10:
            break
    return audio, audio_name, video, video_name, video_gt


def generate_random_cuts(video_path, annotation_per_film_path, num_frames, num_cuts, output_path, idx,
                         type_of_visual_cut, type_of_audio_cut):
    video_files = [x for x in os.listdir(video_path) if x.endswith(".mp4")]

    audio1, audio_name1, video1, video_name1, video1_gt = select_random_video(video_files, annotation_per_film_path)
    audio2, audio_name2, video2, video_name2, video2_gt = select_random_video(video_files, annotation_per_film_path)
    print(video_name1)
    print(video_name2)

    frames1 = video.get_frames(os.path.join(video_path, video1), 224, 224)
    frames2 = video.get_frames(os.path.join(video_path, video2), 224, 224)

    sr1, audio1_data = wavfile.read(os.path.join(video_path, audio1))
    sr2, audio2_data = wavfile.read(os.path.join(video_path, audio2))
    fps1 = video.get_fps(os.path.join(video_path, video1))
    fps2 = video.get_fps(os.path.join(video_path, video2))

    df = pd.DataFrame(columns=["Video1", "Video2", "Shot1", "Shot2", "Type of Visual Cut", "Type of Audio Cut"])
    if not os.path.exists(os.path.join(output_path + "/Synthetic/", "annotations.csv")):
        df.to_csv(os.path.join(output_path + "/Synthetic/", "annotations.csv"), index=False,
                  header=["Video1", "Video2", "Shot1", "Shot2", "Type of Visual Cut", "Type of Audio Cut"])

    for i in range(num_cuts):
        start_frame1, end_frame1, v_shot1, a_shot1 = select_random_shot(video1_gt, num_frames, frames1, video_path, audio1, audio1_data, sr1)
        start_frame2, end_frame2, v_shot2, a_shot2 = select_random_shot(video2_gt, num_frames, frames2, video_path, audio2, audio2_data, sr2)

        if random.choice([True, False]):
            v_shot1, v_shot2 = v_shot2, v_shot1
            a_shot1, a_shot2 = a_shot2, a_shot1

        syn = synthetic_data.SyntheticAudioVisualData(v_shot1, v_shot2, num_frames, a_shot1, a_shot2, fps1, fps2, sr1, sr2)
        transition_frame, shot, audio = syn.generate_transition(type_of_visual_cut, type_of_audio_cut)
        print(shot.shape)
        video.array_to_video(shot, 10, output_path + "/Synthetic/Video/" + str(idx) + ".mp4")
        if isinstance(audio, np.ndarray):
            wavfile.write(output_path + "/Synthetic/Audio/" + str(idx) + ".wav", sr1, audio)
        else:
            audio.export(output_path + "/Synthetic/Audio/" + str(idx) + ".wav", format="wav")
        new_row = [video_name1, video_name2, (start_frame1, start_frame1 + transition_frame),
                   (start_frame2, start_frame2 + num_frames-transition_frame),
                   type_of_visual_cut, type_of_audio_cut]
        df.loc[len(df)] = new_row
        idx += 1
    df.to_csv(os.path.join(output_path + "/Synthetic/", "annotations.csv"), mode="a", index=False, header=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('video_path', type=str, help='Path with the video files')
    parser.add_argument('annotation_path', type=str, help="Path with the annotations per video")
    parser.add_argument('output_path', type=str, help="Path with the annotations per video")
    parser.add_argument('num_frames', type=int, help="Number of frames per cut")
    parser.add_argument('N', type=int, help="Number of generated samples")
    args = parser.parse_args()

    type_of_visual_cuts = ["Gradual", "Hard", "No Transition", "Hard", "Gradual", "Hard"]
    type_of_audio_cuts = ["Gradual", "Hard", "No Transition", "No Transition", "Hard", "Gradual"]
    counter = 0

    for visual_cut, audio_cut in zip(type_of_visual_cuts, type_of_audio_cuts):
        print(visual_cut, audio_cut)
        for i in range(args.N//len(type_of_visual_cuts)):
            generate_random_cuts(video_path=args.video_path,
                                 annotation_per_film_path=args.annotation_path,
                                 num_frames=args.num_frames, num_cuts=1, output_path=args.output_path,
                                 idx=counter,
                                 type_of_visual_cut=visual_cut,
                                 type_of_audio_cut=audio_cut)

            counter += 1
