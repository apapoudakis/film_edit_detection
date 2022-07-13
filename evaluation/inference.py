import torch
from utils.video import get_frames


def run(model, test_video, output_path, num_frames, overlap):
    """
    Predict cuts for a given video and write them on a file

    :param model:
    :param test_video:
    :param output_path:
    :param num_frames:
    :param overlap:
    :return:
    """
    frames = get_frames(test_video)
    video_segments = []
    for i in range(0, frames.shape[0]-num_frames, overlap):
        video_segments.append(torch.FloatTensor(frames[i:i+num_frames, :, :, :]))

    inputs = torch.stack(video_segments, dim=0).reshape(-1, 3, 16, 64, 64)
    outputs = model(inputs)
    _, preds = torch.max(outputs.data, 1)
    print(preds.shape)

    results = []
    print(preds)
    print(preds.shape[0])
    i = 0
    while i < preds.shape[0]:
        if preds[i] != 0:
            label = preds[i]
            begin = i
            i += 1
            while i < preds.shape[0] and preds[i] == preds[i-1]:
                i += 1
            end = i-1
            results.append((begin * overlap + 1, end * overlap + 16 + 1, label.item()))

        else:
            i += 1

    with open(output_path, 'w') as fp:
        for item in results:
            # write each item on a new line
            fp.write(str(item) + "\n")

    return results


