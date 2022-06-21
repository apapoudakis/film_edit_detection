"""
B&W film data of the Media Ecology Project
https://mediaecology.dartmouth.edu/wp/
"""
import os
from utils import video

dataset_path = "../../Data/RedHenLab/Color Films/"


def filtering():
    """
    Filter B&W films data based on if they contain audio data
    """

    filtered_films_path = os.path.join(dataset_path, "EditedColorFilms")

    if not os.path.exists(filtered_films_path):
        os.mkdir(filtered_films_path)

    for film in os.listdir(dataset_path):
        if film.endswith(".mp4"):
            film_name = film.split(".")[0]
            video.audio_check(os.path.join(dataset_path, film), save_path=os.path.join(filtered_films_path, film_name))


if __name__ == "__main__":
    filtering()
