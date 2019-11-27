import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torchvision
import torch
import torch.nn.functional as F


class Network(NetworkBase):
    def __init__(self, input_dim, input_chann=1, output_dim=64):
        super(Network, self).__init__()
        self._name = 'image_to_hand_joints'
        self.fc0 = nn.Linear(2048, 128)
        self.fc1 = nn.Linear(128 + input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 312)
        self.fc4 = nn.Linear(312, 256)
        self.fcHR_residual = nn.Linear(256, 128)
        self.fcHR_residual_2 = nn.Linear(128, 45)
        self.fcR = nn.Linear(256, 64)
        self.fcR_2 = nn.Linear(64, 3)
        self.fcT = nn.Linear(256, 64)
        self.fcT_2 = nn.Linear(64, 3)

    def forward(self, image_representation, hand_representations, Ro, Ts, moreinfo):
        _B = len(hand_representations)
        x_image = self.fc0(image_representation)
        x = self.fc1(torch.cat((x_image, hand_representations, Ro, Ts, moreinfo), -1))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)

        x_hr = self.fcHR_residual(x)
        x_hr = F.relu(x_hr)
        HR = self.fcHR_residual_2(x_hr) + hand_representations

        x_r = self.fcR(x)
        x_hr = F.relu(x_hr)
        R = Ro + self.fcR_2(x_r)

        x_t = self.fcT(x)
        x_t = F.relu(x_t)
        T = Ts + self.fcT_2(x_t)
        return HR, R, T
