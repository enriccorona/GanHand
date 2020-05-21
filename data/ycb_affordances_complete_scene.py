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

from utils.data_utils import fast_load_obj

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

        assert opt.batch_size == 1

        # read dataset
        self._read_dataset_paths()
        self.noise_units = 12
        self.loader = pil_loader

        self._dataset_size = len(self.indexs_split)

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        # FORCE IMAGE FROM VIDEO 5
        imgname = self.data_dir + '/data/YCB_Video_Dataset/' + self.imgnames[self.indexs_split[index]]
        video, frame = imgname.split('/')[-2:]
        video = int(video)
        frame = int(frame.split('-')[0])

        plane = self.planes[self.indexs_split[index]]

        if video < 60:
            intrinsics = [[1066.778, 0, 312.9869], [0, 1067.487, 241.3109], [0, 0, 1]]
        else:
            intrinsics = [[1077.836, 0, 323.7872], [0, 1078.189, 279.6921], [0, 0, 1]]

        fullsize_img = self.loader(imgname.split('\n')[0])

        # NOTE: ALSO DIVIDES BY 256:
        img = resize(fullsize_img, (256, 256))
        img = img - self.means_rgb
        img = img / self.std_rgb


## LOAD GT OBJECTS AND GRASPS FOR SCENE:
        meta = scipy.io.loadmat(imgname.split('-color')[0] + '-meta.mat')
        gt_object_ids = meta['cls_indexes'][:, 0] - 1

        all_gt_obj_verts = []
        all_gt_obj_faces = []
        all_gt_resampled_verts = []
        gt_grasp_repr = []
        gt_grasp_trans = []
        gt_grasp_taxonomy = []

        # LOADING PRECOMPUTED AVAILABLE GRASPS!!!
        available_repr = np.load(self.data_dir + '/data/YCB_Affordance_grasps/mano_representation_%d.npy'%(self.indexs_split[index]), allow_pickle=True)
        available_trans = np.load(self.data_dir + '/data/YCB_Affordance_grasps/mano_translation_%d.npy'%(self.indexs_split[index]), allow_pickle=True)
        available_taxonomies = np.load(self.data_dir + '/data/YCB_Affordance_grasps/mano_taxonomy_%d.npy'%(self.indexs_split[index]), allow_pickle=True)

        # Get list of all obj poses (Vertices and faces!)
        for i in range(len(gt_object_ids)):
            objpose = meta['poses'][:, :, i]
            p = np.matmul(objpose[:3,0:3], self.obj_verts[gt_object_ids[i]].T) + objpose[:3,3].reshape(-1,1)
            resampled_p = np.matmul(objpose[:3,0:3], self.resampled_objects_800verts[gt_object_ids[i]].T) + objpose[:3,3].reshape(-1,1)

            all_gt_obj_verts.append(p.T)
            all_gt_resampled_verts.append(resampled_p.T)
            all_gt_obj_faces.append(self.obj_faces[gt_object_ids[i]])

            if len(available_repr[i]) == 0:
                continue

            grasp_ind = np.random.randint(0, len(available_repr[i]))
            gt_grasp_repr.append(available_repr[i][grasp_ind])
            gt_grasp_trans.append(available_trans[i][grasp_ind])
            gt_grasp_taxonomy.append(available_taxonomies[i][grasp_ind])


## LOAD POSE PREDICTED BY POSE ESTIMATION METHOD
        predPose = np.load(self.data_dir + '/segmentation-driven-pose/predictions/pred_'+str(self.indexs_split[index])+'.npy', allow_pickle=True)
        all_obj_verts = []
        all_obj_faces = []
        obj_bboxes = []
        obj_poses = []
        allmasks = []
        # TODO: Get list of all obj poses (Vertices and faces!)
        object_ids = predPose[:, 0]
        resampled_objects = []

        for i in range(len(object_ids)):
            objpose = predPose[i, 1]
            p = np.matmul(objpose[:3,0:3], self.obj_verts[object_ids[i]].T) + objpose[:3,3].reshape(-1,1)

            all_obj_verts.append(p.T)
            all_obj_faces.append(self.obj_faces[object_ids[i]])

            dense_verts = np.matmul(objpose[:3, 0:3], self.densely_resampled_obj_verts[object_ids[i]].T).T + objpose[:3, 3]
            points800 = np.matmul(objpose[:3, 0:3], self.resampled_objects_800verts[object_ids[i]].T).T + objpose[:3, 3]

            vp = ycb_utils.vertices_final_projection(dense_verts, intrinsics)
            vp = np.int32(vp)

            # Object sometimes has a part outside the image:
            vp[vp[:, 1]<0, 1] = 0
            vp[vp[:, 0]<0, 0] = 0
            vp[vp[:, 1]>479, 1] = 479
            vp[vp[:, 0]>639, 0] = 639

            maskImg = np.zeros_like(fullsize_img[:, :, 0], np.uint8)
            maskImg[vp[:, 1], vp[:, 0]] = 255

            maskImg = (resize(maskImg, (256, 256))>0.02)*1

            allmasks.append(maskImg)
            resampled_objects.append(points800)
            obj_bboxes.append(np.concatenate((dense_verts.max(0), dense_verts.min(0))))
            obj_poses.append(objpose[:3,0:3])

        img = np.float32(img)

        noise = np.random.normal(0, 1.0, (self.noise_units, 1, 1))
        noise = noise.repeat(256, 1).repeat(256, 2)

        grasp_repr = 0
        grasp_trans = 0

        # pack data
        sample = {'rgb_img': img,
                  'mask_img': allmasks,
                  'noise_img': noise,
                  'object_id': -1,
                  'plane_eq': plane,
                  'taxonomy': gt_grasp_taxonomy,
                  'hand_gt_representation': gt_grasp_repr,
                  'hand_gt_translation': gt_grasp_trans,
                  '3d_points_object': all_obj_verts,
                  '3d_faces_object': all_obj_faces,
                  'id_object': object_ids,
                  'pose_object': obj_poses,
                  'bbox_object': obj_bboxes,
                  'object_points_resampled': resampled_objects,
                  '3d_points_resampled_gt': all_gt_resampled_verts,
                  '3d_points_object_gt': all_gt_obj_verts,
                  '3d_faces_object_gt': all_gt_obj_faces,
                  '3d_classes_object_gt': gt_object_ids,
                  'fullsize_imgs': fullsize_img,
                  'camera_intrinsics': intrinsics,
                  }

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):

        self.imgnames = np.load(self.data_dir + '/data/imgnames.npy')
        #self.imgnames = np.load('/tmp-network/fast/ecorona/predictions/YCB-Video-Out_per_image/imgnames.npy')

        if self._mode == 'test':
            #filestest = np.loadtxt('/tmp-network/fast/ecorona/data/YCB_Video_Dataset/image_sets/keyframe.txt', dtype='str')
            #indexs_test = []
            #for i in range(len(filestest)):
            #    newind = np.where(self.imgnames == ('/tmp-network/fast/ecorona/data/YCB_Video_Dataset/data/%s-color.png\n'%filestest[i]))[0][0]
            #    indexs_test.append(newind)
            #np.save('indexs_test_set.npy', indexs_test)
            indexs_test_files = np.load('indexs_test_set.npy')
            self.indexs_split = np.arange(len(self.imgnames))
            self.indexs_split = self.indexs_split[indexs_test_files]
        else:
            self.indexs_split = np.arange(len(self.imgnames))

        #models = glob.glob('/tmp-network/fast/ecorona/data/YCB_Video_Dataset/models/0*')
        #models = glob.glob('/tmp-network/fast/ecorona/data/YCB_objects/0*/google_16k/model_watertight_1000def.obj')

        models = glob.glob(self.data_dir + '/data/models/0*/google_16k/textured.obj')
        models.sort()
        objects_in_YCB = np.load(self.data_dir + '/data/objects_in_YCB.npy')
        models = np.array(models)[objects_in_YCB]
        offset_ycbs = np.load(self.data_dir + '/data/offsets.npy')
        #offset_ycbs = np.load('/tmp-network/fast/ecorona/data/YCB_objects/translation_between_YCBObjects_and_YCBVideos.npy')

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
        #self.densely_resampled_obj_verts = np.load('densely_resampled_objects.npy') #densely_resampled
        self.resampled_objects_800verts = np.load('resampled_objects_800verts.npy')

        #self.planes = np.load('/tmp-network/fast/ecorona/data/YCB_Video_Dataset/allplanes.npy')
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
                    data_type.append(torch.DoubleTensor(args[j][key]))
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
