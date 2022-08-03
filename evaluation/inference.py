import utils.video
from helpers import segments_generator
import torchvision.transforms
import torch
from models import deep_SBD


def run(model, threshold, test_video, output_path, num_frames, overlap):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])

    t = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        normalize,
    ])

    frames = utils.video.get_frames(test_video)
    frames = torch.FloatTensor(frames)
    frames = frames.reshape(-1, 3, 64, 64)
    normalized_frames = torch.zeros_like(frames)
    for j in range(frames.shape[0]):
        normalized_frames[j, :, :, :] = t(frames[j, :, :, :])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    iterator = segments_generator(normalized_frames, num_frames, overlap)
    temp = []
    idx = 0
    for segment in iterator:
        segment = segment.to(device)
        segment = segment.reshape(-1, 3, 64, 64)

        output = model(segment.reshape(1, 3, 16, 64, 64))
        _, preds = torch.max(output.data, 1)
        prediction = preds.item()
        temp.append(prediction)

        idx += 1

    print(temp)
    print(len(temp))
    i = 0

    results = []
    while i < len(temp):
        if temp[i] != 0:
            label = temp[i]
            begin = i
            i += 1
            while i < len(temp) and temp[i] == temp[i - 1]:
                i += 1
            end = i - 1

            first_frame = frames[begin * (num_frames // 2), :, :, :]
            last_frame = frames[end * (num_frames // 2) + 16, :, :, :]

            # post-processing to decrease false positives (FP)
            if deep_SBD.post_processing(first_frame.cpu().numpy(), last_frame.cpu().numpy(), threshold):
                print("***")
                continue

            results.append((begin * (num_frames // 2), end * (num_frames // 2) + 16, label))
        else:
            i += 1

    with open(output_path, 'w') as fp:
        for item in results:
            fp.write(str(item) + "\n")

    return results
