from utils import video
from PIL import Image


def get_concat_h_blank(im1, im2, color=(0, 0, 0)):

    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def concatenate_multiple_images(imgs):
    _im = imgs.pop(0)
    for img in imgs:
        _im = get_concat_h_blank(_im, img)

    return _im


def visualize_video(video_path):
    """
    Visualize a sequence of frames by concatenating them
    """

    frames = video.get_frames(video_path, 224, 224)
    images = []
    for f in frames:
        im = Image.fromarray(f)
        images.append(im)

    concat_frames = concatenate_multiple_images(images[8:])
    concat_frames.show()
