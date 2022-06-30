from utils import video
import numpy as np
import random


class SyntheticVideoData:

    """ Create synthetic visual cut between two shots"""

    def __init__(self, video_segment1, video_segment2, N):
        """
        :param video_segment1: frames of shot1
        :param video_segment2: frames of shot2
        :param N: transition duration (frames)
        """
        self.video_segment1 = video_segment1
        self.video_segment2 = video_segment2
        self.N = N

    def generate_transition(self, type_of_cut):

        if type_of_cut == "Gradual":

            gradual_window_size = random.choice(list(range(6, 16)))
            frames = np.zeros_like(self.video_segment1)
            frames[:self.N // 2 - gradual_window_size // 2, :, :, :] = self.video_segment1[
                                                                       :self.N // 2 - gradual_window_size // 2, :, :, :]

            # Draw uniform distributed samples and sort them in descending order
            alpha = np.random.uniform(0, 1, gradual_window_size).tolist()
            alpha.sort(reverse=True)
            for i, index in enumerate(range(self.N // 2 - gradual_window_size // 2, self.N // 2 + gradual_window_size // 2)):
                frames[index, :, :, :] = video.frames_composition(self.video_segment1[index, :, :, :],
                                                                  self.video_segment2[index, :, :, :], alpha[i])

            frames[self.N // 2 + gradual_window_size // 2:, :, :, :] = self.video_segment2[
                                                                  self.N // 2 + gradual_window_size // 2:, :, :, :]
            return frames
        elif type_of_cut == "Abrupt":
            frames = np.concatenate(
                (self.video_segment1[self.N // 2:, :, :, :], self.video_segment2[:self.N // 2, :, :, :]), axis=0)
            return frames
        else:
            print("Unknown Type!")


class SyntheticAudioData:
    pass
