from utils import transitions, audio_transitions
import random


class SyntheticVideoData:
    """ Create synthetic visual cut between two shots"""

    def __init__(self, video_segment1, video_segment2, N):
        """
        :param video_segment1: shot1 (N frames)
        :param video_segment2: shot2 (N frames)
        :param N: number of frames of the generated video
        """
        self.video_segment1 = video_segment1
        self.video_segment2 = video_segment2
        self.N = N

    def generate_transition(self, type_of_cut):

        if type_of_cut == "Gradual":
            return transitions.gradual_cut(self.video_segment1, self.video_segment2, self.N)

        elif type_of_cut == "Hard":
            transition_frame = random.randint(2, self.N - 2)
            shot1 = self.video_segment1[:transition_frame, :, :, :]
            shot2 = self.video_segment2[: self.N - transition_frame, :, :, :]

            return transitions.abrupt_cut(shot1, shot2)

        elif type_of_cut == "No Transition":
            return random.choice([self.video_segment1, self.video_segment2])
        else:
            print("Unknown Type!")


class SyntheticAudioData:
    """ Create synthetic audio transitions between two shots"""

    def __init__(self, audio_segment1, audio_segment2, N):
        """
        :param audio_segment1:
        :param audio_segment2:
        :param N:
        """
        self.audio_segment1 = audio_segment1
        self.audio_segment2 = audio_segment2
        self.N = N

    def generate_transition(self, type_of_cut):
        if type_of_cut == "Hard":
            return audio_transitions.abrupt_cut(self.audio_segment1, self.audio_segment2)
        elif type_of_cut == "No Transition":
            return random.choice([self.audio_segment2, self.audio_segment2])
        elif type_of_cut == "Gradual":
            return audio_transitions.gradual_cut(self.audio_segment1, self.audio_segment2)
        else:
            print("Unknown Type!")



