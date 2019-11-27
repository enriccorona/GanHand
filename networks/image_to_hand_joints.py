import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torchvision
import torch

class Network(NetworkBase):
    def __init__(self, output_dim=64):
        super(Network, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.fc = nn.Linear(2048, output_dim)
        #self.model = self.model.cuda()

        self._name = 'image_to_hand_joints'

    def forward(self, x):
        #from IPython import embed
        #embed()
        # replicate spatially and concatenate domain information
        #c = c.unsqueeze(2).unsqueeze(3)
        #c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        #x = torch.cat([x, c], dim=1)
        #features = self.main(x)
        return self.model(x)
        #return #self.img_reg(features), self.attetion_reg(features)

        #x = self.model.conv1(x)
        #x = self.model.bn1(x)
        #x = self.model.relu(x)
        #x = self.model.maxpool(x)
#        x = self.model.layer1(x)
#        x = self.model.layer2(x)
#        x = self.model.layer3(x)
#        x = self.model.layer4(x)
#        x = self.avgpool(x)
        #x = self.model.fc(x)
