import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torchvision
import torch
import torch.nn.functional as F

# Obtained from Atlasnet repo > auxiliary/model.py

class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 2500):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        batchsize = x.size()[0]
        # print(x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class Atlasnet(NetworkBase):
    def __init__(self, num_points = 2500, bottleneck_size = 512, nb_primitives = 125, pretrained_encoder = True, cuda=True):
        super(Atlasnet, self).__init__()
        self.usecuda = cuda
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.nb_primitives = nb_primitives
        self.pretrained_encoder = pretrained_encoder
        #self.encoder = torchvision.models.resnet18(pretrained=self.pretrained_encoder, num_classes=1024)
        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size = 2 +self.bottleneck_size) for i in range(0, self.nb_primitives)])

    def forward(self, x):
        x = x[:,:3,:,:].contiguous()
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = torch.cuda.FloatTensor(x.size(0), 2, self.num_points//self.nb_primitives)
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

    def decode(self, x):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = torch.cuda.FloatTensor(x.size(0), 2, self.num_points//self.nb_primitives)
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y.type_as(rand_grid)), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

    def forward_inference(self, x, grid):
        x = self.encoder(x)
        outs = []
        for i in range(0, self.nb_primitives):
            if self.usecuda:
                rand_grid = torch.cuda.FloatTensor(grid[i])
            else:
                rand_grid = torch.FloatTensor(grid[i])

            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

    def forward_inference_from_latent_space(self, x, grid):
        outs = []
        for i in range(0, self.nb_primitives):
            rand_grid = torch.cuda.FloatTensor(grid[i])
            rand_grid = rand_grid.transpose(0, 1).contiguous().unsqueeze(0)
            rand_grid = rand_grid.expand(x.size(0), rand_grid.size(1), rand_grid.size(2)).contiguous()
            # print(rand_grid.sizerand_grid())
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat( (rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        return torch.cat(outs, 2).contiguous().transpose(2,1).contiguous()

