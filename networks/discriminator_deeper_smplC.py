import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch.nn.functional as F

class Discriminator(NetworkBase):
    """Discriminator. PatchGAN."""
    def __init__(self, input_size=45):
        super(Discriminator, self).__init__()
        self._name = 'discriminator_smplC'
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        return x
