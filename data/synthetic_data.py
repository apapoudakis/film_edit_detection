import pandas as pd
from utils import video
import os
import numpy as np
import matplotlib.pyplot as plt
import random


class SyntheticVideoData:

    def __init__(self, data_path, annotation_file, N, height, width):
        self.data = pd.read_csv(annotation_file)
        self.data_path = data_path
        self.N = N
        self.height = height
        self.width = width

    def generate_transition(self, type_of_transition):
        vid1_index = random.randrange(len(self.data))
        vid2_index = random.randrange(len(self.data))
        video1_path = os.path.join(self.data_path, self.data["Film"][vid1_index] + ".mp4")
        video2_path = os.path.join(self.data_path, self.data["Film"][vid2_index] + ".mp4")

        frames1 = video.get_frames(video1_path, self.height, self.width)
        frames2 = video.get_frames(video2_path, self.height, self.width)
        reference_frame1, _ = video.timecode_to_frame(video1_path, self.data["Timecode"][vid1_index])
        reference_frame2, _ = video.timecode_to_frame(video2_path, self.data["Timecode"][vid2_index])

        start_frame1 = random.choice(list(range(0, reference_frame1 - 2*self.N)) + list(range(reference_frame1, len(frames1) - 2*self.N)))
        end_frame1 = start_frame1 + self.N // 2
        start_frame2 = random.choice(list(range(0, reference_frame2 - 2*self.N)) + list(range(reference_frame2, len(frames2) - 2*self.N)))
        end_frame2 = start_frame2 + self.N // 2

        video_segment1 = frames1[start_frame1: end_frame1, :, :, :]
        video_segment2 = frames2[start_frame2: end_frame2, :, :, :]
        print(video_segment2.shape)
        print(video_segment1.shape)

        # Check the type of transition
        if type_of_transition == "Gradual":
            gradual_window_size = random.choice(list(range(6, 16)))
            print("Gradual Window Size: ", gradual_window_size)
            frames = np.concatenate((video_segment1, video_segment2), axis=0)
            gradual_list_indices = range(self.N // 2 - gradual_window_size // 2, self.N // 2 + gradual_window_size // 2)
            alpha = 1
            for i, index in enumerate(gradual_list_indices):
                alpha = alpha - 0.05
                frames[index, :, :, :] = video.frames_composition(video_segment1[self.N // 2 - gradual_window_size + i],
                                                                  video_segment2[i], alpha)
            print(frames.shape)
            for i in range(frames.shape[0]):
                plt.imshow(frames[i, :, :, :])
                plt.pause(0.001)
            plt.close()
        elif type_of_transition == "Abrupt":
            frames = np.concatenate((video_segment1, video_segment2), axis=0)
            print(frames.shape)

            for i in range(frames.shape[0]):
                plt.imshow(frames[i, :, :, :])
                plt.pause(0.001)
            plt.close()
        else :
            print("hi")


if __name__ == "__main__":

    annotation_file = "../../Data/RedHenLab/Color Films/EditedColorFilms/ColorFilmsAnnotations.csv"
    videos_path = "../../Data/RedHenLab/Color Films/EditedColorFilms/"

    syn = SyntheticVideoData(videos_path, annotation_file, 64, 128, 128)
    syn.generate_transition("Gradual")

