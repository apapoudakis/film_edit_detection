import numpy as np
import random
import cv2


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


def abrupt_cut(shot1, shot2):
    return np.concatenate((shot1, shot2), axis=0)


def gradual_cut(shot1, shot2, N):
    transition_frame = random.randint(0, N - 6)
    gradual_window_size = random.choice(list(range(6, N-transition_frame+1)))

    # Draw uniform distributed samples and sort them in descending order
    alpha = np.random.uniform(0, 1, gradual_window_size).tolist()
    alpha.sort(reverse=True)

    frames = np.zeros_like(shot1)

    frames[:transition_frame] = shot1[:transition_frame]
    for i, index in enumerate(range(transition_frame, transition_frame + gradual_window_size)):
        frames[index, :, :, :] = frames_composition(shot1[index, :, :, :], shot2[i, :, :, :], alpha[i])

    frames[transition_frame+gradual_window_size:, :, :, :] = shot2[gradual_window_size:N-transition_frame, :, :, :]

    return frames
