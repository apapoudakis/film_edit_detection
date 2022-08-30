import utils.video
import torchvision.transforms
import torch
from models import deep_SBD
import utils.audio


def segments_generator(frames, num_frames, overlap):
    for i in range(0, len(frames) - num_frames, overlap):
        yield frames[i:i + num_frames, :, :, :].permute(1, 0, 2, 3)


def run(model, threshold, test_video, output_path, num_frames, overlap, width, height):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])
    t = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
        normalize,
    ])

    frames = utils.video.get_frames(test_video, width=width, height=height)
    temp_frames = torch.FloatTensor(frames)
    temp_frames = temp_frames.reshape(-1, 3, width, height)
    normalized_frames = torch.zeros_like(temp_frames)
    for j in range(temp_frames.shape[0]):
        normalized_frames[j, :, :, :] = t(temp_frames[j, :, :, :])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    iterator = segments_generator(normalized_frames, num_frames, overlap)
    temp = []
    idx = 0
    for segment in iterator:
        segment = segment.to(device)
        segment = segment.reshape(-1, 3, width, height)

        output = model(segment.reshape(1, 3, 16, width, height))
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
            if deep_SBD.post_processing(first_frame, last_frame, threshold):
                print("Prediction Removed")
                continue

            results.append((begin * (num_frames // 2), end * (num_frames // 2) + 16, label))
        else:
            i += 1

    with open(output_path, 'w') as fp:
        for item in results:
            fp.write(str(item) + "\n")

    return results
