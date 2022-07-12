import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from utils import video
import pickle
import torchvision.transforms
from models import deep_SBD, train
import torch.nn as nn
import torch
from tqdm import tqdm
from evaluation.metrics import accuracy


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
                frames = video.get_frames(os.path.join(data_path, row["Type of Cut"], str(i) + ".mp4"))
                frames = torch.FloatTensor(frames)
                frames = frames.reshape(-1, 3, height, width)
                #
                for j in range(frames.shape[0]):
                    frames[j, :, :, :] = t(frames[j, :, :, :])

                torch.save(frames, os.path.join(data_path, "video_samples/", str(i) + ".pt"))

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

        sample = torch.load(self.videos_stack[index])
        return sample.permute(1, 0, 2, 3), self.labels[index]


syn_data = VideoDataset("../../../Data/RedHenLab/Color Films/EditedColorFilms/SyntheticDataset/",
                        "../../../Data/RedHenLab/Color Films/EditedColorFilms/SyntheticDataset/annotations.csv",
                        64, 64)

video_loader = DataLoader(syn_data, batch_size=8, shuffle=True)

model = deep_SBD.Model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#
train_size = int(0.9 * len(syn_data))
test_size = len(syn_data) - train_size

print("Training Size", train_size)
print("Testing Size", test_size)

train_dataset, test_dataset = torch.utils.data.random_split(syn_data, [train_size, test_size])
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
# train.train_loop(model, train_dataset, test_dataset, 16, 10, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
# model = model.to(device)

train_losses = []
test_losses = []

train.train_loop(model, train_dataset, test_dataset, batch_size=8, num_epochs=10, device=device)


# for epoch in range(10):
#     model.train()
#     train_epoch_losses = []
#     test_epoch_losses = []
#     for it, data in enumerate(tqdm(train_loader)):
#         x, labels = data
#         x = x.to(device)
#         labels = labels.to(device)
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward
#         outputs = model(x)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         print("Accuracy:", accuracy(outputs, labels))
#         train_epoch_losses.append(loss.item())
#
#     with torch.no_grad():
#         for test_it, data in enumerate(tqdm(test_loader)):
#             test_x, test_labels = data
#             test_x = test_x.to(device)
#             test_labels = test_labels.to(device)
#
#             test_outputs = model(test_x)
#             test_loss = criterion(test_outputs, test_labels)
#             test_epoch_losses.append(test_loss.item())
#
#     train_losses.append(sum(train_epoch_losses)/len(train_epoch_losses))
#     test_losses.append(sum(test_epoch_losses)/len(test_epoch_losses))
#
#     print(f"Epoch: {epoch+1} Train Loss: {train_losses[epoch]} Test Loss: {test_losses[epoch]}")
#
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.plot(train_losses)
# plt.plot(test_losses)
# plt.show()
