"""
deepSBD model https://arxiv.org/abs/1705.03281
Implementations is from https://github.com/Tangshitao/ClipShots_basline
"""
import torch.nn as nn
import cv2
import utils.video


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv3d(3, 96, kernel_size=3, stride=(1, 2, 2),
                               padding=(0, 0, 0), bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=0)
        self.conv2 = nn.Conv3d(96, 256, kernel_size=3, stride=(1, 2, 2),
                               padding=(0, 0, 0), bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=0)
        self.conv3 = nn.Conv3d(256, 384, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv3d(384, 384, kernel_size=(3, 3, 3), stride=1,
                               padding=1, bias=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv3d(384, 256, kernel_size=(3, 3, 3), stride=1,
                               padding=1, bias=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=0)
        self.fc6 = nn.Linear(18432, 2048)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc7 = nn.Linear(2048, 2048)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc8 = nn.Linear(2048, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.fc8(x)

        return x


def post_processing(img1, img2, threshold):
    hist1 = utils.video.colour_histogram(img1)
    hist2 = utils.video.colour_histogram(img2)

    distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    print(distance)
    if distance < threshold:
        return True

    return False
