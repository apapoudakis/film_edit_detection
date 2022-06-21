from moviepy.editor import VideoFileClip
import moviepy
import os
import cv2
import numpy as np
import ffmpeg


def audio_check(video, save_path=None):
    """
    Checks if a video contains audio data
    """

    video_name = os.path.basename(video).split(".")[0]
    audio = moviepy.editor.VideoFileClip(video).audio

    if all(audio.max_volume(stereo=True) == 0):
        return False

    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        audio.write_audiofile(save_path + "/" + video_name + "_audio.wav")

    return True


def get_frames(video, width=64, height=64):
    """
    Video to Numpy Array
    (https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md#convert-video-to-numpy-array)

    :param video:
    :param width:
    :param height:
    """

    out, _ = (
        ffmpeg.input(video).output('pipe:', format='rawvideo', pix_fmt='rgb24',
                                   s='{}x{}'.format(width, height)).run(capture_stdout=True,
                                                                        capture_stderr=True))
    video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])

    return video


def timecode_to_frame(video, timecode):
    """
    Converts timecode to frame index

    :param video: path of the video file
    :param timecode: HH:MM:SS:FF
    """
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)

    splitted_timecode = timecode.split(":")
    cut_time = int(splitted_timecode[0]) * 3600 + int(splitted_timecode[1]) * 60 + int(splitted_timecode[2])
    transition_frame = int(fps * cut_time) + int(splitted_timecode[3])

    return transition_frame, cut_time
