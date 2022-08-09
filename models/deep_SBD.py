"""
deepSBD model https://arxiv.org/abs/1705.03281
"""
import torch.nn as nn
import cv2
import utils.video


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=96, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
        self.conv1_bn = nn.BatchNorm3d(96)  # instead of Local Response Normalization (LRN)
        self.max_pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(in_channels=96, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 2, 2))
        self.conv2_bn = nn.BatchNorm3d(256)  # instead of Local Response Normalization (LRN)
        self.max_pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2))
        self.conv3 = nn.Conv3d(in_channels=256, out_channels=384, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(in_channels=384, out_channels=384, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv5 = nn.Conv3d(in_channels=384, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.max_pool5 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fc6 = nn.Linear(8*7*7*256, 2048)
        self.fc7 = nn.Linear(2048, 2048)
        self.fc8 = nn.Linear(2048, 3)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.conv1_bn(self.conv1(x)))
        h = self.max_pool1(h)

        h = self.relu(self.conv2_bn(self.conv2(h)))
        h = self.max_pool2(h)

        h = self.relu(self.conv3(h))
        h = self.relu(self.conv4(h))
        h = self.relu(self.conv5(h))
        h = self.max_pool5(h)

        h = h.view(h.size(0), -1)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)
        logits = self.fc8(h)

        return logits


def post_processing(img1, img2, threshold):
    hist1 = utils.video.colour_histogram(img1)
    hist2 = utils.video.colour_histogram(img2)

    distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    print(distance)
    if distance < threshold:
        return True

    return False
