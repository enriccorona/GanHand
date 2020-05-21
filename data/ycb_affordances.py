import os.path
from data.dataset import DatasetBase
from PIL import Image
import random
import numpy as np
import pickle
import os
import torch
import scipy.io

import point_cloud_utils as pcu

import os.path
from data.dataset import DatasetBase
import random
import numpy as np
import pickle
import os
from utils import cv_utils
from skimage.transform import resize
import torch
from utils import util
import torchvision
from PIL import Image

import glob
from utils import ycb_utils

from .dataset_real_imgs import fast_load_obj

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return np.array(img)

class Dataset(DatasetBase):
    def __init__(self, opt, mode):
        super(Dataset, self).__init__(opt, mode)
        self._name = 'Dataset_obman'
        self.data_dir = opt.data_dir

        # read dataset
        self._read_dataset_paths()
        self.noise_units = 12
        self.loader = pil_loader

        self._dataset_size = len(self.indexs_split)

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        # FORCE IMAGE FROM VIDEO 5
        #index = np.random.randint(8059, 9564)
        #index = np.random.randint(71086, 73534)

        imgname = self.imgnames[self.indexs_split[index]]
        #video, frame = str.split(imgname, '/')[-2:]
        video, frame = imgname.split('/')[-2:]
        video = int(video)
        #frame = int(str.split(frame, '-')[0])
        frame = int(frame.split('-')[0])

        plane = self.planes[self.indexs_split[index]]

        if video < 60:
            intrinsics = [[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]]
        else:
            intrinsics = [[1077.836, 0, 323.7872], [0, 1078.189, 279.6921], [0, 0, 1]]

        #fullsize_img = self.loader(str.split(imgname, '\n')[0])
        fullsize_img = self.loader(imgname.split('\n')[0])
        maskImg = np.zeros_like(fullsize_img[:, :, 0], np.uint8)

        # NOTE: ALSO DIVIDES BY 256:
        img = resize(fullsize_img, (256, 256))
        img = img - self.means_rgb
        img = img / self.std_rgb

        # THIS FOR TESTING:
        if self._mode == 'test': #or self._mode == 'val':
            predPose = np.load(self.data_dir + '/segmentation-driven-pose/predictions/pred_'+str(self.indexs_split[index])+'.npy', allow_pickle=True)
            all_obj_verts = []
            all_obj_faces = []
            # TODO: Get list of all obj poses (Vertices and faces!)
            object_ids = predPose[:, 0]
            for i in range(len(object_ids)):
                objpose = predPose[i, 1]
                p = np.matmul(objpose[:3,0:3], self.obj_verts[object_ids[i]].T) + objpose[:3,3].reshape(-1,1)

                all_obj_verts.append(p.T)
                all_obj_faces.append(self.obj_faces[object_ids[i]])

            # TODO: WRITE DATASET CODE FOR GETTING ALL OF THESE. I can just provide all possible hands, with list of objects and let model iterate (in test) or choose one (in train)
            ind = np.random.randint(0, len(object_ids))

            grasp_repr = 0
            grasp_trans = 0

        else:
            # THIS FOR TRAINING:
            #meta = scipy.io.loadmat( str.split(imgname, '-color')[0] + '-meta.mat')
            meta = scipy.io.loadmat(imgname.split('-color')[0] + '-meta.mat')
            object_ids = meta['cls_indexes'][:, 0] - 1

            all_obj_verts = []
            all_obj_verts_resampled800 = []
            all_obj_faces = []

            # Get list of all obj poses (Vertices and faces!)
            for i in range(len(object_ids)):
                objpose = meta['poses'][:, :, i]
                p = np.matmul(objpose[:3,0:3], self.obj_verts[object_ids[i]].T) + objpose[:3,3].reshape(-1,1)
                points800 = np.matmul(objpose[:3, 0:3], self.resampled_objects_800verts[object_ids[i]].T).T + objpose[:3, 3]

                all_obj_verts.append(p.T)
                all_obj_faces.append(self.obj_faces[object_ids[i]])
                all_obj_verts_resampled800.append(points800)

            # LOADING PRECOMPUTED AVAILABLE GRASPS!!!
            available_repr = np.load(self.data_dir + '/data/YCB_Affordance_grasps/mano_representation_%d.npy'%(self.indexs_split[index]), allow_pickle=True)
            available_trans = np.load(self.data_dir + '/data/YCB_Affordance_grasps/mano_translation_%d.npy'%(self.indexs_split[index]), allow_pickle=True)
            available_taxonomies = np.load(self.data_dir + '/data/YCB_Affordance_grasps/mano_taxonomy_%d.npy'%(self.indexs_split[index]), allow_pickle=True)

            while True:
                ind = np.random.randint(0, len(object_ids))
                if len(available_repr[ind]) > 0:
                    grasp_ind = np.random.randint(0, len(available_repr[ind]))
                    break

            grasp_repr = available_repr[ind][grasp_ind]
            grasp_trans = available_trans[ind][grasp_ind]
            grasp_taxonomy = available_taxonomies[ind][grasp_ind]

            dense_verts = np.matmul(meta['poses'][:3, 0:3, ind], self.densely_resampled_obj_verts[object_ids[ind]].T).T+meta['poses'][:3, 3, ind]

        vp = ycb_utils.vertices_final_projection(dense_verts, intrinsics)
        #vp = ycb_utils.vertices_final_projection(all_obj_verts[ind], intrinsics)
        vp = np.int32(vp)

        # Object sometimes has a part outside the image:
        vp[vp[:, 1]<0, 1] = 0
        vp[vp[:, 0]<0, 0] = 0
        vp[vp[:, 1]>479, 1] = 479
        vp[vp[:, 0]>639, 0] = 639

        maskImg[vp[:, 1], vp[:, 0]] = 255

        maskImg = (resize(maskImg, (256, 256))>0.05)*1

        img = np.float32(img)

        noise = np.random.normal(0, 1.0, (self.noise_units, 1, 1))
        noise = noise.repeat(256, 1).repeat(256, 2)

        # pack data
        sample = {'rgb_img': img,
                  'mask_img': maskImg,
                  'noise_img': noise,
                  'object_id': ind,
                  'plane_eq': plane,
                  'taxonomy': grasp_taxonomy-1,
                  'hand_gt_representation': grasp_repr,
                  'hand_gt_translation': grasp_trans,
                  '3d_points_object': all_obj_verts,
                  '3d_faces_object': all_obj_faces,
                  'id_object': object_ids,
                  'object_points_resampled': all_obj_verts_resampled800,
                  'fullsize_imgs': fullsize_img,
                  'camera_intrinsics': intrinsics,
                  }

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):
        if self._mode == 'test':
            raise('ERROR, this file is not yet prepare to run this!!! Maybe you can run on ycb_real_complete_scene')

        self.imgnames = np.load(self.data_dir + '/data/imgnames.npy')

        #filestest = np.loadtxt('/tmp-network/fast/ecorona/data/YCB_Video_Dataset/image_sets/keyframe.txt', dtype='str')
        #indexs_test = []
        #for i in range(len(filestest)):
        #    newind = np.where(self.imgnames == ('/tmp-network/fast/ecorona/data/YCB_Video_Dataset/data/%s-color.png\n'%filestest[i]))[0][0]
        #    indexs_test.append(newind)
        #np.save('indexs_test_set.npy', indexs_test)

        if self._mode == 'train':
            indexs_files = np.load('indexs_train_set.npy')
            self.indexs_split = np.arange(len(self.imgnames))
            self.indexs_split = self.indexs_split[indexs_files]
        elif self._mode == 'val':
            indexs_files = np.load('indexs_val_set.npy')
            self.indexs_split = np.arange(len(self.imgnames))
            self.indexs_split = self.indexs_split[indexs_files]

        models = glob.glob(self.data_dir + '/data/models/0*/google_16k/textured.obj')
        models.sort()
        objects_in_YCB = np.load(self.data_dir + '/data/objects_in_YCB.npy')
        models = np.array(models)[objects_in_YCB]
        offset_ycbs = np.load(self.data_dir + '/data/offsets.npy')

        obj_verts = []
        densely_resampled = []
        obj_faces = []
        for i in range(len(models)):
            obj = fast_load_obj(open(models[i], 'rb'))[0]
            #obj = fast_load_obj(open(models[i]+'/model_watertight_1000def.obj', 'rb'))[0]
            #obj = fast_load_obj(open(models[i]+'/model_watertight_2000def.obj', 'rb'))[0]
#            densely_resampled.append(pcu.sample_mesh_lloyd(obj['vertices'], np.int32(obj['faces']), 20000))
            obj_verts.append(obj['vertices'] - offset_ycbs[i])
            obj_faces.append(obj['faces'])

        self.obj_verts = obj_verts
        self.obj_faces = obj_faces
        self.densely_resampled_obj_verts = np.load('densely_resampled_objects.npy') 
        self.resampled_objects_800verts = np.load('resampled_objects_800verts.npy') 

        self.planes = np.load(self.data_dir + '/data/plane_equations.npy')

        # Resnet norms
        self.means_rgb = [0.485, 0.456, 0.406]
        self.std_rgb = [0.229, 0.224, 0.225]

    def _read_ids(self, file_path, extension):
        files = os.listdir(file_path)
        files = [f.replace(extension, '') for f in files]
        return files

    def collate_fn(self, args):
        length = len(args)
        keys = list(args[0].keys())
        data = {}

        for i, key in enumerate(keys):
            data_type = []

            if key == 'rgb_img' or key == 'mask_img' or key == 'noise_img' or key == 'plane_eq' or key == 'hand_gt_representation' or key == 'hand_gt_translation':
                for j in range(length):
                    data_type.append(torch.FloatTensor(args[j][key]))
                data_type = torch.stack(data_type)
            elif key == 'label' or key == 'taxonomy':
                labels = []
                for j in range(length):
                    labels.append(args[j][key])
                data_type = torch.LongTensor(labels)
            else:
                for j in range(length):
                    data_type.append(args[j][key])
            data[key] = data_type

        return data
