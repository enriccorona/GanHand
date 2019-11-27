import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch.nn.functional as F

class Discriminator(NetworkBase):
    """Discriminator. PatchGAN."""
    def __init__(self, input_size=45):
        super(Discriminator, self).__init__()
        self._name = 'discriminator_smplh'
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        return x
