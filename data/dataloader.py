"""
Audio Visual Dataloader inspired by this implementation
https://github.com/iariav/End-to-End-VAD
"""

import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from scipy.io import wavfile
from utils import video
import pickle
import torchvision.transforms


class AudioDataset(Dataset):
    def __init__(self, data_path, annotation_file, time_window):
        self.audio_files = os.listdir(data_path)
        self.annotation_data = pd.read_csv(annotation_file)
        self.data_path = data_path
        self.labels = []

        if not os.path.exists(os.path.join(data_path, "audio.pkl")):
            temp_film = self.annotation_data["Film"][0]
            sample_rate, audio_data = wavfile.read(os.path.join(data_path,
                                                                self.annotation_data["Film"][0] + "_audio.wav"))
            counter = 0
            for i, row in self.annotation_data.iterrows():

                if temp_film != row["Film"]:
                    temp_film = row["Film"]
                    sample_rate, audio_data = wavfile.read(
                        os.path.join(data_path, row["Film"] + "_audio.wav"))
                cut_time = video.timecode_to_secs(row["Timecode"])

                segment_audio = audio_data[(cut_time - time_window // 2) * sample_rate: (
                                                                                                cut_time + time_window // 2) * sample_rate,
                                :]
                if segment_audio.shape[0] != (time_window * sample_rate): continue

                wavfile.write(data_path + "audio_samples/" + str(counter) + ".wav", sample_rate, segment_audio)
                counter += 1

                if row["Type"] == "soft":
                    temp_label = 1
                else:
                    temp_label = 2
                self.labels.append(temp_label)

            with open(data_path + "audio.pkl", 'wb') as f:
                pickle.dump(self.labels, f)

        # labels = open(audio_path + "audio.pkl", 'rb')
        with open(data_path + "audio.pkl", 'rb') as f:
            self.labels = pickle.load(f)
        self.counter = len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        idx_path = self.data_path + "audio_samples/" + str(index) + ".wav"
        sample_rate, audio_data = wavfile.read(idx_path)
        sample = torch.from_numpy(audio_data).type(torch.FloatTensor)

        return sample, self.labels[index]


class VideoDataset(Dataset):
    def __init__(self, data_path, annotation_file, time_window, N, width, height):
        self.annotation_data = pd.read_csv(annotation_file)
        self.data_path = data_path
        self.N = N
        self.labels = []
        self.videos_stack = []

        if not os.path.exists(os.path.join(data_path, "video.pkl")):

            if not os.path.exists(os.path.join(data_path, "video_samples")):
                os.mkdir(os.path.join(data_path, "video_samples"))

            temp_film = self.annotation_data["Film"][0]
            frames = video.get_frames(os.path.join(data_path, temp_film + ".mp4"), width=width, height=height)
            sample_rate, audio_data = wavfile.read(os.path.join(data_path, self.annotation_data["Film"][0] + "_audio"
                                                                                                             ".wav"))
            counter = 0
            for i, row in self.annotation_data.iterrows():
                print(i)
                if temp_film != row["Film"]:
                    temp_film = row["Film"]
                    sample_rate, audio_data = wavfile.read(
                        os.path.join(data_path, row["Film"] + "_audio.wav"))
                    frames = video.get_frames(os.path.join(data_path, temp_film + ".mp4"), width=width, height=height)
                cut_time = video.timecode_to_secs(row["Timecode"])

                segment_audio = audio_data[(cut_time - time_window // 2) * sample_rate:
                                           (cut_time + time_window // 2) * sample_rate, :]

                if segment_audio.shape[0] != (time_window * sample_rate):
                    continue
                normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])

                t = torchvision.transforms.Compose([
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.ToTensor(),
                    normalize,
                ])

                reference_frame, cut_time = video.timecode_to_frame(os.path.join(data_path, temp_film + ".mp4"),
                                                                    row["Timecode"])
                key_frames = frames[reference_frame - N // 2:reference_frame + N // 2, :, :, :]
                key_frames = torch.FloatTensor(key_frames)
                key_frames = key_frames.reshape(-1, 3, height, width)

                # for j in range(N):
                #     key_frames[j, :, :, :] = t(key_frames[j, :, :, :])

                torch.save(key_frames,
                           "../../Data/RedHenLab/Color Films/EditedColorFilms/video_samples/" + str(counter) + ".pt")
                counter += 1

                if row["Type"] == "soft":
                    temp_label = 1
                else:
                    temp_label = 2
                self.labels.append(temp_label)

            with open(data_path + "video.pkl", 'wb') as f:
                pickle.dump(self.labels, f)

        with open(data_path + "video.pkl", 'rb') as f:
            self.labels = pickle.load(f)

        for i in range(len(self.labels)):
            self.videos_stack.append(
                torch.load("../../Data/RedHenLab/Color Films/EditedColorFilms/video_samples/" + str(i) + ".pt"))

    def __len__(self):
        return len(self.videos_stack)

    def __getitem__(self, index):
        sample = self.videos_stack[index]
        return sample, self.labels[index]


class AudioVideoDataset(Dataset):
    def __init__(self, data_path, annotation_file, time_window, N, width, height):
        self.audio_dataset = AudioDataset(data_path=data_path, annotation_file=annotation_file,
                                          time_window=time_window)
        self.video_dataset = VideoDataset(data_path=data_path, annotation_file=annotation_file,
                                          time_window=time_window, N=N, height=height, width=width)

    def __len__(self):
        return len(self.audio_dataset)

    def __getitem__(self, index):
        audio_sample, audio_label = self.audio_dataset.__getitem__(index)
        video_sample, video_label = self.video_dataset.__getitem__(index)
        av_label = audio_label | video_label
        return audio_sample, video_sample, av_label


if __name__ == "__main__":

    video_data = VideoDataset("../../Data/RedHenLab/Color Films/EditedColorFilms/",
                              "../../Data/RedHenLab/Color Films/EditedColorFilms/ColorFilmsAnnotations.csv",
                              time_window=8,
                              width=64, height=64, N=64)

    video_loader = DataLoader(video_data, batch_size=8, shuffle=True)

    for frames, x in video_loader:
        print(frames.shape)
        print(x.shape)
        print(x)
