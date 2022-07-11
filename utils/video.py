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


# def get_frames(video, width=64, height=64):
#     """
#     Video to Numpy Array
#     (https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md#convert-video-to-numpy-array)
#
#     :param video:
#     :param width:
#     :param height:
#     """
#
#     out, _ = (
#         ffmpeg.input(video).output('pipe:', format='rawvideo', pix_fmt='rgb24',
#                                    s='{}x{}'.format(width, height)).run(capture_stdout=True,
#                                                                         capture_stderr=True))
#     video = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
#
#     return video


def get_frames(video, width=64, height=64):
    frames = []
    cap = cv2.VideoCapture(video)
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            frames.append(cv2.resize(img, (height, width)))
    video = np.stack(frames, axis=0)  # dimensions (T, H, W, C)
    return video


def timecode_to_secs(timecode):
    """
    :param timecode: HH:MM:SS:FF
    :return:
    """
    splitted_timecode = timecode.split(":")
    time = int(splitted_timecode[0]) * 3600 + int(splitted_timecode[1]) * 60 + int(splitted_timecode[2])
    return time


def timecode_to_frame(video, timecode):
    """
    Converts timecode to frame index

    :param video: path of the video file
    :param timecode: HH:MM:SS:FF
    """
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cut_time = timecode_to_secs(timecode)
    splitted_timecode = timecode.split(":")
    transition_frame = int(fps * cut_time) + int(splitted_timecode[3])

    return transition_frame, cut_time


def frames_composition(img1, img2, alpha):
    """
    I(x) = alpha *img1(x) + (1.0-alpha)*img2
    as defined by https://arxiv.org/abs/1705.03281

    :param img1
    :param img2
    :param alpha
    """

    beta = 1.0 - alpha
    comp_img = cv2.addWeighted(img1, alpha, img2, beta, 0.0)
    return comp_img


def array_to_video(frames, fps, output_path):
    """
    Save numpy array as video file

    :param frames:
    :param fps:
    :param output_path:
    :return:
    """

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames.shape[2], frames.shape[1]), True)
    for i in range(frames.shape[0]):
        out.write(cv2.cvtColor(frames[i, :, :, :], cv2.COLOR_BGR2RGB))
    out.release()
