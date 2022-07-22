import os
import random
from utils import annotation_utils, video
from scipy.io import wavfile
import re
import synthetic_data


def select_random_shot(video_gt, num_frames, frames, video_path, audio, audio_data, sr):
    while 1:
        idx1 = random.randrange(3, len(video_gt))
        begin1 = int(video_gt[idx1].split(" ")[0])
        end1 = int(video_gt[idx1].split(" ")[1])
        start_frame = random.randint(begin1, end1)
        fps = int(video.get_fps(os.path.join(video_path, audio.split(".wav")[0] + ".mp4")))
        if (start_frame + num_frames) < end1 \
                and frames[start_frame:start_frame + num_frames, :, :, :].shape[0] == num_frames \
                and audio_data[(start_frame * sr // fps):(start_frame + num_frames) * sr // fps, :].shape[0] > 0:
            visual_shot = frames[start_frame:start_frame + num_frames, :, :, :]
            audio_shot = audio_data[(start_frame * sr // fps):(start_frame + num_frames) * sr // fps, :]
            break
    return visual_shot, audio_shot


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


def generate_random_cuts(video_path, annotation_per_film_path, num_frames, num_cuts, output_path, idx, type_of_cut):

    video_files = [x for x in os.listdir(video_path) if x.endswith(".mp4")]
    audio_files = [x for x in os.listdir(video_path) if x.endswith(".wav")]

    audio1, audio_name1, video1, video_name1, video1_gt = select_random_video(video_files, annotation_per_film_path)
    audio2, audio_name2, video2, video_name2, video2_gt = select_random_video(video_files, annotation_per_film_path)
    print(video_name1)
    print(video_name2)

    frames1 = video.get_frames(os.path.join(video_path, video1), 64, 64)
    frames2 = video.get_frames(os.path.join(video_path, video2), 64, 64)

    sr1, audio1_data = wavfile.read(os.path.join(video_path, audio1))
    sr2, audio2_data = wavfile.read(os.path.join(video_path, audio2))

    for i in range(num_cuts):
        v_shot1, a_shot1 = select_random_shot(video1_gt, num_frames, frames1, video_path, audio1, audio1_data, sr1)
        v_shot2, a_shot2 = select_random_shot(video2_gt, num_frames, frames2, video_path, audio2, audio2_data, sr1)

        if random.choice([True, False]):
            v_shot1, v_shot2 = v_shot2, v_shot1
            a_shot1, a_shot2 = a_shot2, a_shot1

        syn = synthetic_data.SyntheticAudioVisualData(v_shot1, v_shot2, num_frames, a_shot1, a_shot2)
        shot, audio = syn.generate_transition(type_of_cut)
        print(shot.shape)
        print(audio.shape)


if __name__ == "__main__":

    generate_random_cuts(video_path="../../../Data/TRECVID/temp",
                         annotation_per_film_path="../../../Data/TRECVID/msb",
                         num_frames=16, num_cuts=1, output_path="./",
                         idx=0,
                         type_of_cut="No Transition")
