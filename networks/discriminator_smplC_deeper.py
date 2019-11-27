import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch.nn.functional as F

class Discriminator(NetworkBase):
    """Discriminator. PatchGAN."""
    def __init__(self, input_size=45):
        super(Discriminator, self).__init__()
        self._name = 'discriminator_smplC'
        self.fc1 = nn.Linear(input_size, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
