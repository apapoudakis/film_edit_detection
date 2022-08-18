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

    def generate_transition(self, type_of_cut, transition_frame=None, gradual_window_size=None):

        if type_of_cut == "Gradual":
            return transitions.gradual_cut(self.video_segment1, self.video_segment2, self.N, transition_frame, gradual_window_size)

        elif type_of_cut == "Hard":
            if transition_frame is None:
                transition_frame = random.randint(2, self.N-1)
            shot1 = self.video_segment1[:transition_frame, :, :, :]
            shot2 = self.video_segment2[: self.N - transition_frame, :, :, :]

            return transitions.abrupt_cut(shot1, shot2)

        elif type_of_cut == "No Transition":
            return self.video_segment1
            # return random.choice([self.video_segment1, self.video_segment2])
        else:
            print("Unknown Type!")


class SyntheticAudioData:
    """ Create synthetic audio transitions between two shots"""

    def __init__(self, audio_segment1, audio_segment2, N, fps1, fps2, sr1, sr2):
        """
        :param audio_segment1:
        :param audio_segment2:
        :param N:
        """
        self.audio_segment1 = audio_segment1
        self.audio_segment2 = audio_segment2
        self.N = N
        self.fps1 = fps1
        self.fps2 = fps2
        self.sr1 = sr1
        self.sr2 = sr2

    def generate_transition(self, type_of_cut, transition_frame=None, gradual_window_size=None):
        if type_of_cut == "Hard":
            if transition_frame is None:
                transition_frame = random.randint(2, self.N-1)
            print(self.fps1)
            print(self.sr1)
            audio_segment1 = self.audio_segment1[:(transition_frame+100) * int(self.sr1) // int(self.fps1), :]
            audio_segment2 = self.audio_segment2[: (self.N - transition_frame+100) * int(self.sr2) // int(self.fps2), :]

            return audio_transitions.abrupt_cut(audio_segment1, audio_segment2)
        elif type_of_cut == "No Transition":
            return self.audio_segment1
            # return random.choice([self.audio_segment1, self.audio_segment2])
        elif type_of_cut == "Gradual":
            print(self.audio_segment1.shape)
            print(self.audio_segment2.shape)
            return audio_transitions.gradual_cut(self.audio_segment1, self.audio_segment2, self.sr1,
                                                 self.sr2, self.fps1, self.fps2, self.N, transition_frame)
        else:
            print("Unknown Type!")


class SyntheticAudioVisualData:
    """ Create synthetic audio transitions between two shots"""

    def __init__(self, video_segment1, video_segment2, num_frames, audio_segment1, audio_segment2, fps1, fps2, sr1, sr2):
        self.visual_dataset = SyntheticVideoData(video_segment1, video_segment2, num_frames)
        self.audio_dataset = SyntheticAudioData(audio_segment1, audio_segment2, num_frames, fps1, fps2, sr1, sr2)
        self.num_frames = num_frames

    def generate_transition(self, type_of_visual_cut, type_of_audio_cut):
        transition_frame = random.randint(2, self.num_frames-1)
        shot = self.visual_dataset.generate_transition(type_of_visual_cut, transition_frame)
        audio = self.audio_dataset.generate_transition(type_of_audio_cut, transition_frame)

        return transition_frame, shot, audio

