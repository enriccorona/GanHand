import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torchvision
import torch
import torch.nn.functional as F


class Network(NetworkBase):
    def __init__(self, input_chann=1, output_dim=64):
        super(Network, self).__init__()
        self._name = 'image_to_hand_joints'
        self.fc1 = nn.Linear(45 + 3*2 + 3, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fcHR_residual = nn.Linear(128, 45)
        self.fcR = nn.Linear(128, 3)
        self.fcT = nn.Linear(128, 3)

    def forward(self, hand_representations, Rs, Ts):
        _B = len(hand_representations)
        x = self.fc1(torch.cat((hand_representations, Rs.view(_B, -1), Ts), -1))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        HR = self.fcHR_residual(x) + hand_representations
        R = self.fcR(x)
        T = self.fcT(x)
        return HR, R, T
