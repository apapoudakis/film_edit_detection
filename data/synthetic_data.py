from utils import video
import numpy as np
import random


class SyntheticVideoData:

    def __init__(self, video_segment1, video_segment2, N):
        self.video_segment1 = video_segment1
        self.video_segment2 = video_segment2
        self.N = N

    def generate_transition(self, type_of_cut):

        if type_of_cut == "Gradual":

            gradual_window_size = random.choice(list(range(6, 16)))
            frames = np.zeros_like(self.video_segment1)
            alpha = 1

            frames[:self.N // 2 - gradual_window_size // 2, :, :, :] = self.video_segment1[
                                                                       :self.N // 2 - gradual_window_size // 2, :, :, :]
            for i in range(self.N // 2 - gradual_window_size // 2, self.N // 2 + gradual_window_size // 2):
                frames[i, :, :, :] = video.frames_composition(self.video_segment1[i, :, :, :],
                                                              self.video_segment2[i, :, :, :], alpha)
                alpha = alpha - random.uniform(0.05, 0.2)
                if alpha < 0:
                    alpha = 0
            frames[self.N // 2 + gradual_window_size // 2:, :, :, :] = self.video_segment2[
                                                                       self.N // 2 + gradual_window_size // 2:, :, :, :]
            return frames
        elif type_of_cut == "Abrupt":
            frames = np.concatenate(
                (self.video_segment1[self.N // 2:, :, :, :], self.video_segment2[:self.N // 2, :, :, :]), axis=0)
            return frames
        else:
            print("Unknown Type!")
