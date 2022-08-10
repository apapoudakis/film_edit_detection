import os
import numpy as np
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from utils import video
import pickle
import torchvision.transforms
from models import deep_SBD, train
import torch.nn as nn
import torch


class VideoDataset(Dataset):

    def __init__(self, data_path, annotation_file, width, height):
        self.annotation_data = pd.read_csv(annotation_file)
        self.data_path = data_path
        self.labels = []
        self.videos_stack = []

        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])

        t = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            # torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

        if not os.path.exists(os.path.join(data_path, "video.pkl")):

            if not os.path.exists(os.path.join(data_path, "video_samples")):
                os.mkdir(os.path.join(data_path, "video_samples"))

            counter = 0
            for i, row in self.annotation_data.iterrows():
                frames = video.get_frames(os.path.join(data_path, row["Type of Cut"], str(row["Idx"]) + ".mp4"), width,
                                          height)
                frames = torch.FloatTensor(frames)
                frames = frames.reshape(-1, 3, height, width)

                for j in range(frames.shape[0]):
                    frames[j, :, :, :] = t(frames[j, :, :, :])

                # torch.save(frames, os.path.join(data_path, "video_samples/", str(i) + ".pt"))
                with open(os.path.join(data_path, "video_samples/", str(row["Idx"]) + ".pt"), 'wb') as f:
                    pickle.dump(frames, f)

                if row["Type of Cut"] == "No Transition":
                    temp_label = 0
                elif row["Type of Cut"] == "Hard":
                    temp_label = 1
                else:
                    temp_label = 2
                self.labels.append(temp_label)

            with open(data_path + "video.pkl", 'wb') as f:
                pickle.dump(self.labels, f)

        with open(data_path + "video.pkl", 'rb') as f:
            self.labels = pickle.load(f)

        for i in range(len(self.labels)):
            self.videos_stack.append(os.path.join(data_path, "video_samples/", str(i) + ".pt"))

    def __len__(self):
        return len(self.videos_stack)

    def __getitem__(self, index):

        # sample = torch.load(self.videos_stack[index])
        with open(self.videos_stack[index], 'rb') as f:
            sample = pickle.load(f)
        return sample.permute(1, 0, 2, 3), self.labels[index]


print("hi")
syn_data = VideoDataset("../../../Data/deepSBD/EditedDeepSBD/",
                        "../../../Data/deepSBD/EditedDeepSBD/annotations.csv",
                        64, 64)
print("hi2")

# test_data = VideoDataset("../../../Data/RAI/segments/",
#                          "../../../Data/RAI/segments/annotations.csv",
#                          64, 64)

train_size = int(0.9 * len(syn_data))
test_size = len(syn_data) - train_size

print("Training Size", train_size)
print("Testing Size", test_size)

train_dataset, test_dataset = torch.utils.data.random_split(syn_data, [train_size, test_size])
test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = deep_SBD.Model()
# model.load_state_dict(torch.load("model_final.pt"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = model.to(device)
model.eval()

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

train.train_loop(model, train_dataset, test_dataset, batch_size=8, num_epochs=5, device=device)
