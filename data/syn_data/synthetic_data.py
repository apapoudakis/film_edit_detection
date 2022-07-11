import pandas as pd
import os
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

        if type_of_cut == "Dissolve":

            gradual_window_size = random.choice(list(range(6, 16)))
            frames = np.zeros_like(self.video_segment1)
            frames[:self.N // 2 - gradual_window_size // 2, :, :, :] = self.video_segment1[
                                                                       :self.N // 2 - gradual_window_size // 2, :, :, :]

            # Draw uniform distributed samples and sort them in descending order
            alpha = np.random.uniform(0, 1, gradual_window_size).tolist()
            alpha.sort(reverse=True)
            for i, index in enumerate(
                    range(self.N // 2 - gradual_window_size // 2, self.N // 2 + gradual_window_size // 2)):
                frames[index, :, :, :] = video.frames_composition(self.video_segment1[index, :, :, :],
                                                                  self.video_segment2[index, :, :, :], alpha[i])

            frames[self.N // 2 + gradual_window_size // 2:, :, :, :] = self.video_segment2[
                                                                       self.N // 2 + gradual_window_size // 2:, :, :, :]
            return frames
        elif type_of_cut == "Hard":
            frames = np.concatenate(
                (self.video_segment1[self.N // 2:, :, :, :], self.video_segment2[:self.N // 2, :, :, :]), axis=0)
            return frames
        elif type_of_cut == "No Transition":
            return random.choice([self.video_segment1, self.video_segment2])
        else:
            print("Unknown Type!")


def generate_random_shots(film_files, annotation_per_film_path, data_path, num_frames, num_cuts, output_path, idx,
                          type_of_cut):
    """
    Select two random shots ensuring that they do not contain cut

    :param num_cuts:
    :param film_files:
    :param annotation_per_film_path:
    :param data_path:
    :param num_frames:
    :return:
    """
    film1 = random.choice(film_files)
    film2 = random.choice(film_files)
    film1_annotations = pd.read_csv(os.path.join(annotation_per_film_path, film1))
    film2_annotations = pd.read_csv(os.path.join(annotation_per_film_path, film2))
    frames1 = video.get_frames(os.path.join(data_path, film1.split(" - ")[0] + ".mp4"), 64, 64)
    frames2 = video.get_frames(os.path.join(data_path, film2.split(" - ")[0] + ".mp4"), 64, 64)
    df = pd.DataFrame(columns=["Film1", "Film2", "Type of Cut"])
    if not os.path.exists(os.path.join(output_path, "annotations.csv")):
        df.to_csv(os.path.join(output_path, "annotations.csv"), index=False)

    for i in range(num_cuts):

        while 1:
            idx1 = random.randrange(3, len(film1_annotations))
            if isinstance(film1_annotations["Timecode"][idx1], float): continue
            try:
                reference_frame1, _ = video.timecode_to_frame(os.path.join(data_path, film1.split(" - ")[0] + ".mp4"),
                                                              film1_annotations["Timecode"][idx1])
                reference_temp, _ = video.timecode_to_frame(os.path.join(data_path, film1.split(" - ")[0] + ".mp4"),
                                                            film1_annotations["Timecode"][idx1 - 1])
            except:
                continue

            if reference_frame1 - reference_temp >= 2 * num_frames:
                shot1 = frames1[reference_frame1 - 2 * num_frames:reference_frame1 - num_frames, :, :, :]
                break
        while 1:
            idx2 = random.randrange(1, len(film2_annotations) - 2)
            if isinstance(film2_annotations["Timecode"][idx2], float): continue
            try:
                reference_frame2, _ = video.timecode_to_frame(os.path.join(data_path, film2.split(" - ")[0] + ".mp4"),
                                                              film2_annotations["Timecode"][idx2])
                reference_temp, _ = video.timecode_to_frame(os.path.join(data_path, film2.split(" - ")[0] + ".mp4"),
                                                            film2_annotations["Timecode"][idx2 + 1])
            except:
                continue

            if reference_temp - reference_frame2 >= 2 * num_frames:
                shot2 = frames2[reference_frame2 + num_frames:reference_frame2 + 2 * num_frames, :, :, :]
                break

        # type_of_cut = random.choice(["Hard", "Dissolve", "No Transition"])
        # type_of_cut = "No Transition"
        # print(type_of_cut)
        new_row = [film1.split(" - ")[0], film2.split(" - ")[0], type_of_cut]
        df.loc[len(df)] = new_row
        if random.choice([True, False]):
            shot1, shot2 = shot2, shot1

        syn = SyntheticVideoData(shot1, shot2, 16)
        frames = syn.generate_transition(type_of_cut)
        video.array_to_video(frames, 25, os.path.join(output_path, type_of_cut, str(idx) + ".mp4"))
        idx = idx + 1
    df.to_csv(os.path.join(output_path, "annotations.csv"), mode="a", index=False, header=False)


if __name__ == "__main__":

    data_path = "../../../Data/RedHenLab/Color Films/EditedColorFilms/"
    annotation_per_film_path = "../../../Data/RedHenLab/Color Films/EditedColorFilms/annotations_per_film"
    film_files = os.listdir(annotation_per_film_path)
    samples_counter = 0

    transitions = ["No Transition", "Hard", "Dissolve"]

    for t in transitions:
        print(t)
        for i in range(10):
            print(i)
            generate_random_shots(film_files, annotation_per_film_path, data_path, 16, 1000,
                                  "../../../Data/RedHenLab/Color Films/EditedColorFilms/SyntheticDataset/",
                                  samples_counter, type_of_cut=t)
            samples_counter = samples_counter + 1000
