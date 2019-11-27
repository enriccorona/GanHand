import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch.nn.functional as F
import torchvision

class Discriminator(NetworkBase):
    """Discriminator. PatchGAN."""
    def __init__(self):
        super(Discriminator, self).__init__()
        self._name = 'discriminator_image'
        self.model = torchvision.models.resnet50(pretrained=True)
        input_chann = 1
        output_dim = 1
        self.model.conv1 = nn.Conv2d(input_chann, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        return self.model(x)
