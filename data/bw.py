"""
B&W film data of the Media Ecology Project
https://mediaecology.dartmouth.edu/wp/
"""
import os
from utils import video
import shutil

dataset_path = "../../Data/RedHenLab/Color Films/"


def filtering():
    """
    Filter B&W films data based on if they contain audio data
    """

    colour_films_path = "../../Data/RedHenLab/Color Films/csv files for color sheets with 2 cut types/"
    annotated_films = [x.split(" - ")[0] + ".mp4" for x in os.listdir(colour_films_path)]

    filtered_films_path = os.path.join(dataset_path, "EditedColorFilms")

    if not os.path.exists(filtered_films_path):
        os.mkdir(filtered_films_path)

    for film in os.listdir(dataset_path):
        if film.endswith(".mp4") and film in annotated_films:
            if video.audio_check(os.path.join(dataset_path, film), save_path=filtered_films_path):
                shutil.copy(os.path.join(dataset_path, film), os.path.join(filtered_films_path, film))


if __name__ == "__main__":
    filtering()
