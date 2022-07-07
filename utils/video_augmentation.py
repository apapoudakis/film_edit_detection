import numpy as np
import cv2


def add_noise(frames):
    """
    Add gaussian noisy to a clip based on this code
    https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
    """

    noisy_frames = np.zeros(frames.shape)

    for i in range(frames.shape[0]):
        row, col, ch = frames[i, :, :, :].shape
        mean = 0
        var = np.random.uniform(0, 1)
        sigma = var ** 0.5
        gauss = np.random.uniform(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = frames[i, :, :, :] + gauss

        noisy_frames[i, :, :, :] = noisy

    return noisy_frames.astype("uint8")


def blurred_shot(frames, kernel_size):
    """
    Create a blurred shot based on given kernel size (odd number)
    """
    blurred_frames = np.zeros(frames.shape)

    for i in range(frames.shape[0]):
        out = cv2.GaussianBlur(frames[i, :, :, :], (kernel_size, kernel_size), cv2.BORDER_DEFAULT)
        blurred_frames[i, :, :, :] = out
    return blurred_frames.astype("uint8")
