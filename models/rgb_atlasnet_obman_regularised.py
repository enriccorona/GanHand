import torch
from collections import OrderedDict
from torch.autograd import Variable
import utils.util as util
import utils.plots as plot_utils
from .models import BaseModel
from networks.networks import NetworksFactory
import os
import numpy as np
import sys
sys.path.append("utils/chamfer_dist/")

import torchvision
from torch import nn

from networks import Atlasnet_obman

#import dist_chamfer as chamfer_cuda

def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    P = (rx.t() + ry - 2*zz)
    return P


def NN_loss(x, y, dim=0):
    dist = pairwise_dist(x, y)
    values, indices = dist.min(dim=dim)
    return values.mean()


def chamfer_cuda(a, b):
    x,y = a,b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2,1))
    yy = torch.bmm(y, y.transpose(2,1))
    zz = torch.bmm(x, y.transpose(2,1))
    diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2,1) + ry - 2*zz)
    return P.min(1)[0], P.min(2)[0]



class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__(opt)
        self._name = 'Model_rgb_to_object'

        # create networks
        self._init_create_networks()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        # init
        self._init_losses()
        self.clusters = np.load('clustersk' + str(self._opt.k) + 'norot.npy')

        self.atlas_loss = Atlasnet_obman.AtlasLoss(
                    atlas_loss='chamfer',

#                    lambda_atlas=0.0,
#                    final_lambda_atlas=0.0625,
#                    trans_weight=0.0625,
#                    scale_weight=0.0625,

                    #lambda_atlas=0.167,
                    #final_lambda_atlas=0.167,
                    #trans_weight=0.167,
                    #scale_weight=0.167,
#                    edge_regul_lambda=0,
#                    lambda_laplacian=0,
#                    laplacian_faces=None,
#                    laplacian_verts=None

                lambda_atlas = 16.7,
                #final_lambda_atlas = 0.167,
                trans_weight= 0, #.167,
                scale_weight= 0, #.167,
                edge_regul_lambda= self._opt.lambda_G_fk,
                lambda_laplacian= self._opt.lambda_G_angles,
                laplacian_faces=self._G.test_faces,
                laplacian_verts=self._G.test_verts,
                )

    def _init_create_networks(self):
        from networks import resnet_obman
        self._encoder = resnet_obman.resnet18(pretrained=True)
        self._encoder = self._encoder.cuda()

        # generator network
        self._G = self._create_generator()
        #self._G.init_weights()
        self._G.cuda()

    def _create_generator(self):
        return NetworksFactory.get_by_name('atlasnet_as_in_obman')

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G*10,
                                             betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])
        self._optimizer_encoder = torch.optim.Adam(self._encoder.parameters(), lr=self._current_lr_G,
                                             betas=[self._opt.G_adam_b1, self._opt.G_adam_b2])
        #self._optimizer_G = torch.optim.SGD(self._G.parameters(), lr=self._current_lr_G)

    def set_epoch(self, epoch):
        self.iepoch = epoch

    def _init_losses(self):
        # init losses G
        self.distChamfer =  chamfer_cuda
        #self.distChamfer =  chamfer_cuda.chamferDist()

    def set_input(self, input):
        self._input_rgb_img = input['rgb_img'].float().cuda()
        self._input_obj_verts = input['3d_points_object']
        self._input_obj_faces = input['3d_faces_object']

        return 

    def set_train(self):
        self._G.train()
        self._encoder.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._encoder.eval()
        self._is_train = False

    # get image paths
    def get_image_paths(self):
        return OrderedDict([]) #('real_img', self._input_real_img_path)])

    def forward(self, keep_data_for_visuals=False):
        self._G.eval()
        self._encoder.eval()
        if not self._is_train:
            img = self._input_rgb_img#[:, 0, :, :].unsqueeze(1)
            img = img.permute(0, 3, 1, 2)
            img_features, _ = self._encoder(img)
            prediction = self._G.forward_inference(img_features)#['objpoints3d']

            normalizing_diagonal = 0.20

            loss = torch.FloatTensor([0]).cuda()
            comparing_shapes = []
            for i in range(self._B):
                obj_vert = torch.FloatTensor(self._input_obj_verts[i]).cuda()
                obj_vert = obj_vert - obj_vert.mean(0, keepdim=True)

                bb_max = obj_vert.max(0)[0]
                bb_min = obj_vert.min(0)[0]

                diagonal = torch.sqrt(((bb_max-bb_min)**2).sum())
                div = diagonal/normalizing_diagonal
                obj_vert = obj_vert / div

                obj_vert = obj_vert.unsqueeze(0)

                prediction_i = prediction.copy()
                prediction_i['objpoints3d'] = prediction_i['objpoints3d'][i].unsqueeze(0)
                #prediction_i['objtrans'] = prediction_i['objtrans'][i].unsqueeze(0)
                #prediction_i['objpointscentered3d'] = prediction_i['objpointscentered3d'][i].unsqueeze(0)
                prediction_i['objfaces'] = prediction_i['objfaces']
                prediction_i['objscale'] = prediction_i['objscale'][i].unsqueeze(0)

                loss = loss + self.atlas_loss.compute_loss(prediction_i, obj_vert)[0]
                comparing_shapes.append(obj_vert[0])

            loss = loss/self._B
            self._loss_g_CD = loss

            if keep_data_for_visuals:
                self.training_gt = comparing_shapes
                self._predicted_obj_verts = prediction['objpoints3d']
                self._gt_verts = comparing_shapes

            return prediction

    def optimize_parameters(self, train_generator=True, keep_data_for_visuals=False):
        if self._is_train:
            # convert tensor to variables
            self._B = self._input_rgb_img.size(0)

            loss_G = self._forward_G(keep_data_for_visuals)
            self._optimizer_G.zero_grad()
            self._optimizer_encoder.zero_grad()
            loss_G.backward()
            self._optimizer_G.step()
            self._optimizer_encoder.step()

    def _forward_G(self, keep_data_for_visuals):
        img = self._input_rgb_img#[:, 0, :, :].unsqueeze(1)
        img = img.permute(0, 3, 1, 2)
        img_features, _ = self._encoder(img)
        prediction = self._G.forward_inference(img_features)#['objpoints3d']

        normalizing_diagonal = 0.20

        loss = torch.FloatTensor([0]).cuda()
        comparing_shapes = []
        for i in range(self._B):
            obj_vert = torch.FloatTensor(self._input_obj_verts[i]).cuda()
            obj_vert = obj_vert - obj_vert.mean(0, keepdim=True)

            bb_max = obj_vert.max(0)[0]
            bb_min = obj_vert.min(0)[0]

            diagonal = torch.sqrt(((bb_max-bb_min)**2).sum())
            div = diagonal/normalizing_diagonal
            obj_vert = obj_vert / div

            obj_vert = obj_vert.unsqueeze(0)

            prediction_i = prediction.copy()
            prediction_i['objpoints3d'] = prediction_i['objpoints3d'][i].unsqueeze(0)
            #prediction_i['objtrans'] = prediction_i['objtrans'][i].unsqueeze(0)
            #prediction_i['objpointscentered3d'] = prediction_i['objpointscentered3d'][i].unsqueeze(0)
            prediction_i['objfaces'] = prediction_i['objfaces']
            prediction_i['objscale'] = prediction_i['objscale'][i].unsqueeze(0)

            loss = loss + self.atlas_loss.compute_loss(prediction_i, obj_vert)[0]
            comparing_shapes.append(obj_vert[0])

        loss = loss/self._B
        self._loss_g_CD = loss

        if keep_data_for_visuals:
            self.training_gt = comparing_shapes
            self.predictions_label = prediction
            self._predicted_obj_verts = prediction['objpoints3d']
            self._gt_verts = comparing_shapes

        # combine losses
        return self._loss_g_CD

    def get_current_errors(self):
        loss_dict = OrderedDict([('g_CD', self._loss_g_CD.cpu().data.numpy()),
                                 ])

        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()

        img = self._input_rgb_img[0].cpu().data.numpy()
        img = (img*[0.229, 0.224, 0.225]+[0.485, 0.456, 0.406])*255
        #visuals['1_inputimg'] = plot_utils.plot_rgb(np.int32(img))
        #visuals['2_geometry'] = plot_utils.plot_3d_verts(self._gt_verts[0].cpu().data.numpy())
        #visuals['3_reconstruction'] = plot_utils.plot_3d_verts(self._predicted_obj_verts[0].cpu().data.numpy())
        #visuals['4_mesh_object'] = plot_utils.plot_hand(self._predicted_obj_verts[0].cpu().data.numpy(), self._G.test_faces)
        visuals['5_together'] = plot_utils.plot_pointsandmesh(self._predicted_obj_verts[0].cpu().data.numpy(), self._G.test_faces, self.training_gt[0].cpu().data.numpy())
        visuals['6_together'] = plot_utils.plot_pointsandmesh(self._predicted_obj_verts[0].cpu().data.numpy(), self._G.test_faces, self.training_gt[0].cpu().data.numpy(), switch_axes=True)

        return visuals

    def save(self, label):
        # save networks
        self._save_network(self._G, 'G', label)
        self._save_network(self._encoder, 'encoder', label)

        torch.save(self._G, self._opt.checkpoints_dir + '/model_G_epoch_'+str(label))
        torch.save(self._encoder, self._opt.checkpoints_dir + '/model_encoder_epoch_'+str(label))

        # save optimizers
        self._save_optimizer(self._optimizer_G, 'G', label)
        self._save_optimizer(self._optimizer_encoder, 'encoder', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._G, 'G', load_epoch)
        self._load_network(self._encoder, 'encoder', load_epoch)

        if self._is_train:
            # load optimizers
            self._load_optimizer(self._optimizer_G, 'G', load_epoch)
            self._load_optimizer(self._optimizer_encoder, 'encoder', load_epoch)

    def update_learning_rate(self):
        # updated learning rate G
        lr_decay_G = self._opt.lr_G / self._opt.nepochs_decay
        self._current_lr_G -= lr_decay_G
        for param_group in self._optimizer_G.param_groups:
            param_group['lr'] = self._current_lr_G*10
        print('update G learning rate: %f -> %f' %  (self._current_lr_G*10 + lr_decay_G, self._current_lr_G*10))

        for param_group in self._optimizer_encoder.param_groups:
            param_group['lr'] = self._current_lr_G
        print('update G learning rate: %f -> %f' %  (self._current_lr_G + lr_decay_G, self._current_lr_G))
