import cv2
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Union
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Conv2DReLU(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False
    ):
        super(Conv2DReLU, self).__init__()

        self.module = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            ),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.module(x)
        return x


class Stem(nn.Module):

    def __init__(self, in_channels):
        super(Stem, self).__init__()

        self.base = nn.Sequential(
            Conv2DReLU(in_channels, 32, kernel_size=3, stride=2),
            Conv2DReLU(32, 32, kernel_size=3),
            Conv2DReLU(32, 64, kernel_size=3, padding=1)
        )
        self.conv0 = Conv2DReLU(64, 96, kernel_size=3, stride=2)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch0 = nn.Sequential(
            Conv2DReLU(160, 64, kernel_size=1),
            Conv2DReLU(64, 96, kernel_size=3)
        )
        self.branch1 = nn.Sequential(
            Conv2DReLU(160, 64, kernel_size=1),
            Conv2DReLU(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            Conv2DReLU(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            Conv2DReLU(64, 96, kernel_size=3)
        )
        self.conv1 = Conv2DReLU(192, 192, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        x = self.base(x)
        x0 = self.conv0(x)
        x1 = self.pool0(x)
        x = torch.cat((x0, x1), 1)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x = torch.cat((x0, x1), 1)
        x0 = self.conv1(x)
        x1 = self.pool1(x)
        x = torch.cat((x0, x1), 1)
        return x


class InceptionA(nn.Module):

    def __init__(self, in_channels):
        super(InceptionA, self).__init__()

        self.branch0 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2DReLU(in_channels, 96, kernel_size=1)
        )
        self.branch1 = Conv2DReLU(384, 96, kernel_size=1)
        self.branch2 = nn.Sequential(
            Conv2DReLU(in_channels, 64, kernel_size=1),
            Conv2DReLU(64, 96, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            Conv2DReLU(in_channels, 64, kernel_size=1),
            Conv2DReLU(64, 96, kernel_size=3, padding=1),
            Conv2DReLU(96, 96, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat((x0, x1, x2, x3), 1)
        return x


class ReductionA(nn.Module):

    def __init__(self, in_channels):
        super(ReductionA, self).__init__()

        self.branch0 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch1 = Conv2DReLU(in_channels, 384, kernel_size=3, stride=2)
        self.branch2 = nn.Sequential(
            Conv2DReLU(in_channels, 192, kernel_size=1),
            Conv2DReLU(192, 224, kernel_size=3, padding=1),
            Conv2DReLU(224, 256, kernel_size=3, stride=2)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat((x0, x1, x2), 1)
        return x


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()

        self.branch0 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv2DReLU(in_channels, 128, kernel_size=1)
        )
        self.branch1 = Conv2DReLU(1024, 384, kernel_size=1)
        self.branch2 = nn.Sequential(
            Conv2DReLU(in_channels, 192, kernel_size=1),
            Conv2DReLU(192, 224, kernel_size=(1, 7), padding=(0, 3)),
            Conv2DReLU(224, 256, kernel_size=(7, 1), padding=(3, 0))
        )
        self.branch3 = nn.Sequential(
            Conv2DReLU(in_channels, 192, kernel_size=1),
            Conv2DReLU(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            Conv2DReLU(192, 224, kernel_size=(7, 1), padding=(3, 0)),
            Conv2DReLU(224, 224, kernel_size=(1, 7), padding=(0, 3)),
            Conv2DReLU(224, 256, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x = torch.cat((x0, x1, x2, x3), 1)
        return x


class ReductionB(nn.Module):

    def __init__(self, in_channels):
        super(ReductionB, self).__init__()

        self.branch0 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(
            Conv2DReLU(in_channels, 192, kernel_size=1),
            Conv2DReLU(192, 192, kernel_size=3, stride=2)
        )
        self.branch2 = nn.Sequential(
            Conv2DReLU(in_channels, 256, kernel_size=1),
            Conv2DReLU(256, 256, kernel_size=(1, 7), padding=(0, 3)),
            Conv2DReLU(256, 320, kernel_size=(7, 1), padding=(3, 0)),
            Conv2DReLU(320, 320, kernel_size=3, stride=2)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat((x0, x1, x2), 1)
        return x


class InceptionC(nn.Module):

    def __init__(self, in_channels):
        super(InceptionC, self).__init__()

        self.branch0 = nn.Sequential(
            nn.AvgPool2d(kernel_size=1),
            Conv2DReLU(in_channels, 256, kernel_size=1)
        )
        self.branch1 = Conv2DReLU(in_channels, 256, kernel_size=1)
        self.conv0 = Conv2DReLU(in_channels, 384, kernel_size=1)
        self.conv1 = Conv2DReLU(384, 256, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = Conv2DReLU(384, 256, kernel_size=(3, 1), padding=(1, 0))
        self.branch2 = nn.Sequential(
            Conv2DReLU(in_channels, 384, kernel_size=1),
            Conv2DReLU(384, 448, kernel_size=(1, 3), padding=(0, 1)),
            Conv2DReLU(448, 512, kernel_size=(3, 1), padding=(1, 0))
        )
        self.conv3 = Conv2DReLU(512, 256, kernel_size=(3, 1), padding=(1, 0))
        self.conv4 = Conv2DReLU(512, 256, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.conv0(x)
        x3 = self.conv1(x2)
        x4 = self.conv2(x2)
        x5 = self.branch2(x)
        x6 = self.conv3(x5)
        x7 = self.conv4(x5)
        x = torch.cat((x0, x1, x3, x4, x6, x7), 1)
        return x


class InceptionV4(nn.Module):

    def __init__(self, in_channels, classes=1):
        super(InceptionV4, self).__init__()

        modules = []
        modules.append(Stem(in_channels))
        for i in range(4):
            modules.append(InceptionA(384))
        modules.append(ReductionA(384))
        for i in range(7):
            modules.append(InceptionB(1024))
        modules.append(ReductionB(1024))
        for i in range(3):
            modules.append(InceptionC(1536))
        self.features = nn.Sequential(*modules)
        self.glob_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, classes)

    def forward(self, x):
        x = self.features(x)
        x = self.glob_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x