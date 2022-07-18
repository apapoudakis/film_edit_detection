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

    constant = 1000

    video = get_frames(test_video)
    results = []
    for j in range(0, len(video), constant):
        frames = video[j:j+constant, :, :, :]
        out = []
        for i in range(0, frames.shape[0]-num_frames, overlap):
            out.append(model(torch.FloatTensor(frames[i:i+num_frames, :, :, :]).reshape(1, 3, 16, 64, 64)))

        outputs = torch.stack(out, dim=0).reshape(-1, 3)
        _, preds = torch.max(outputs.data, 1)

        i = 0
        while i < preds.shape[0]:
            if preds[i] != 0:
                label = preds[i]
                begin = i
                i += 1
                while i < preds.shape[0] and preds[i] == preds[i-1]:
                    i += 1
                end = i-1
                results.append((begin * overlap + j + 1, end * overlap + 16 + j + 1, label.item()))

            else:
                i += 1

    with open(output_path, 'w') as fp:
        for item in results:
            fp.write(str(item) + "\n")

    return results
