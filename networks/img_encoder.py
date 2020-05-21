import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torchvision
import torch

class Network(NetworkBase):
    def __init__(self, input_chann=3, output_dim=64):
        super(Network, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)

        if input_chann != 3:
            pretrained_conv1 = self.model.conv1.weight
            self.model.conv1 = nn.Conv2d(input_chann, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.conv1.weight[:, :3] = pretrained_conv1
            self.model.conv1.weight = torch.nn.Parameter(self.model.conv1.weight)
            self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.model.fc = nn.Linear(2048, output_dim)
        self._name = 'img_encoder'

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        img_representation = x.view(x.size(0), -1)
        x = self.model.fc(img_representation)
        return x, img_representation
