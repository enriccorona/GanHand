import os.path
from data.dataset import DatasetBase
from PIL import Image
import random
import numpy as np
import pickle
import os
from utils import cv_utils
from skimage.transform import resize
import torch
from sklearn.cluster import KMeans
import socket
from scipy import ndimage

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
import socket
from scipy import ndimage
from utils import util
import torchvision
from PIL import Image
import accimage

import glob

from .dataset_real_imgs import fast_load_obj

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return np.array(img)


def getBB(depth):
    objs = ndimage.find_objects((depth > 0)*1)[0]
    x, x2 = objs[0].start, objs[0].stop
    y, y2 = objs[1].start, objs[1].stop

    return x, y, x2, y2


class Dataset(DatasetBase):
    def __init__(self, opt, mode):
        super(Dataset, self).__init__(opt, mode)
        self._name = 'Dataset_obman'
        self.pc_name = socket.gethostname()

        # read dataset
        self._read_dataset_paths()
        self.noise_units = 12
        self.loader = pil_loader

        textures = []
        filetextures = glob.glob('/tmp-network/fast/ecorona/data/background-textures/*')
        #filetextures = os.listdir('/tmp-network/fast/ecorona/data/background-textures/')
        for i in range(len(filetextures)):
            textures.append(self.loader(filetextures[i]))
            #textures.append('/tmp-network/fast/ecorona/data/background-textures/' + self.loader(filetextures[i]))
        self.textures = np.array(textures)

        #from IPython import embed
        #embed()

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        path = self._imgs_rgb_dir + '/' + self._ids[index] + '.jpg'
        mask_path = self._imgs_masks_dir + '/' + self._ids[index] + '.png'
        img = self.loader(path)
        mask = self.loader(mask_path)[:, :, 0]

        texture = self.textures[np.random.randint(len(self.textures))].copy()
        xposition = np.random.randint(np.shape(texture)[0] - 256)
        yposition = np.random.randint(np.shape(texture)[1] - 256)
        img[mask!=100] = texture[xposition:xposition+256, yposition:yposition+256][mask!=100]

        img = np.float32(img)/255

        img = img - self.means_rgb
        img = img / self.std_rgb

        noise = np.random.normal(0, 1.0, (self.noise_units, 1, 1))
        noise = noise.repeat(256, 1).repeat(256, 2)

        # TODO: CONSIDER LOADING ALL THESE AND SAMPLE BEFOREHAND!
        #points_object = 0
        #faces_object = 0

        # For train time, object points were previously obtained and resampled
        points_object, faces_object = self.load_obj_details(index)

        # pack data
        sample = {'rgb_img': img,
                  'coords_3d': self.coords_3d[index],
                  'noise_img': noise,
                  'label': self.labels[index],
                  'cluster': self.clusters[self.labels[index], -48:],
                  'verts_3d': np.float32(self.verts_3d[index]),
                  'coords_3d': self.coords_3d[index],
                  'pca_poses': self.pca_poses[index],
                  #'3d_points_object_resampled': points_object,
                  '3d_points_object': points_object,
                  '3d_faces_object': faces_object,
                  'object_directions': self.obj_normals[index],
                  'object_translation': self.obj_translations[index],
                  'mano_rotation': self.mano_rotation[index],
                  'mano_translation': self.mano_translation[index]
                  }

        return sample

    def __len__(self):
        return self._dataset_size

    def load_obj_details(self, index):
        path = '/tmp-network/fast/ecorona/data/ShapeNetCore.v2/' + self.obj_path[index]
        path = str.split(path, 'model_normalized.obj')[0]
        path = path + 'model_watertight.obj'

        with open(path, 'rb') as f:
            mesh = fast_load_obj(f)[0]

            points = mesh['vertices']
            faces = mesh['faces']

        # Obtained from obman repo: https://github.com/hassony2/obman/blob/master/obman/obman.py#L400
        transform = self.obj_transforms[index]
        points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1)
        points = transform.dot(points.T).T[:, :3]

        return points, faces

    def _read_dataset_paths(self):
        if self._mode == 'test' and self.pc_name == 'tro':
            self._root_dir = '/DATA/ecorona/obman/obman/test/'
        else:
            self._root_dir = os.path.join(self._opt.data_dir, self._mode)
        self._imgs_rgb_dir = os.path.join(self._root_dir, 'rgb_obj')
        self._imgs_masks_dir = '/tmp-network/fast/ecorona/data/original_obman/obman/'+self._mode+'/segm/'
        self._imgs_depth_dir = os.path.join(self._root_dir, 'obtained_depth_objs')
        self._imgs_further_depth_dir = os.path.join(self._root_dir, 'obtained_further_depth_objs')
        self._masks_dir = os.path.join(self._root_dir, 'segm')
        self._meta_dir = os.path.join(self._root_dir, 'processed_meta')

        # read ids
        self._ids = self._read_ids(self._meta_dir, '.pkl')
        self._ids.sort()

        # DEMO:
        #self._ids = self._ids[:200]

        # dataset size
        self._dataset_size = len(self._ids)

        # read imgs
        self._read_meta()

        # read obj characteristics
        self.obj_normals = np.load(self._root_dir + '/obtained_normals_objects.npy')
        self.obj_translations = np.load(self._root_dir + '/obtained_translations_objects.npy')

        # Obtained from train set
        self.means_rgb = [0.485, 0.456, 0.406]
        self.std_rgb = [0.229, 0.224, 0.225]

        k = self._opt.k
        km = KMeans(n_clusters=k)
        rot_importance = 15
        try:
            self.clusters = np.load('clustersk'+str(k)+'norot.npy')
        except:
            from IPython import embed
            embed()
            # For the first time, let's just compute and save clusters from train data
            if self._mode != 'train':
                raise('Get clusters from train data for the first time')

            self.clusters = self.get_clusters_kmeans(k=k, rot_importance=0)
            np.save('clustersk'+str(k)+'norot.npy', self.clusters)

        km.cluster_centers_ = self.clusters
        # Let's give 'rot_importance' more importance to rotation - it's only 3 values vs 45 from hand pose
        self.labels = km.predict(self.pca_poses)

    def _read_ids(self, file_path, extension):
        files = os.listdir(file_path)
        files = [f.replace(extension, '') for f in files]
        return files

    def _read_meta(self):
        coords_3d = []
        coords_2d = []
        pca_poses = []
        verts_3d = []
        hand_pose = []
        obj_transforms = []
        grasp_qualities = []
        mano_rotation = []
        mano_translation = []
        obj_path = []
        for i in range(self._dataset_size):
            filepath = self._meta_dir + '/' + self._ids[i] + '.pkl'
            with open(filepath, 'rb') as f:
                gt = pickle.load(f)
                coords_3d.append(gt['coords_3d'])
                coords_2d.append(gt['coords_2d'])
                pca_poses.append(gt['pca_pose'])
                verts_3d.append(gt['verts_3d'])
                hand_pose.append(gt['hand_pose'])
                obj_transforms.append(gt['affine_transform'])
                grasp_qualities.append(gt['grasp_quality'])
                mano_rotation.append(gt['additional_rot'][:, 0])
                mano_translation.append(gt['additional_trans'])
                obj_path.append(gt['obj_path'][47:])
        self.coords_3d = np.array(coords_3d)
        self.coords_2d = np.array(coords_2d)
        self.pca_poses = np.array(pca_poses)
        self.verts_3d = np.float16(verts_3d)
        self.hand_pose = np.array(hand_pose)
        self.grasp_qualities = np.array(grasp_qualities)
        self.mano_rotation = np.array(mano_rotation)
        self.mano_translation = np.array(mano_translation)
        self.obj_transforms = np.array(obj_transforms)
        self.obj_path = obj_path
        return

    def collate_fn(self, args):
        length = len(args)
        keys = list(args[0].keys())
        data = {}

        for i, key in enumerate(keys):
            data_type = []

            if key == 'rgb_img' or key == 'depth_img' or key == 'noise_img' or key == 'further_depth_img' or key=='object_directions' or key=='object_translation' or key == 'coords_3d' or key == 'mano_rotation' or key == 'mano_translation' or key == 'pca_poses':
                for j in range(length):
                    data_type.append(torch.DoubleTensor(args[j][key]))
                data_type = torch.stack(data_type)
            elif key == 'label':
                labels = []
                for j in range(length):
                    labels.append(args[j][key])
                data_type = torch.LongTensor(labels)
            else:
                for j in range(length):
                    data_type.append(args[j][key])
            data[key] = data_type

        return data

    def get_clusters_kmeans(self, k, rot_importance):
        km = KMeans(n_clusters=k)
        # Let's give 'rot_importance' importance to rotation - it's only 3 values vs 45 from hand pose
        km.fit(self.pca_poses)
        return km.cluster_centers_


# GET PERCENTAGE OF POINTS THAT ARE SEEN FROM INPUT IMAGE:
    def get_percentage_seen_points(self):
        percentages = []
        from utils import contactutils
        for index in range(self._dataset_size):
            points_object, faces_object = self.load_obj_details(index)
            obj_triangles = points_object[faces_object].copy()

            points_object[:, 2] -= 0.01
            exterior = contactutils.batch_mesh_contains_points(
                torch.FloatTensor(points_object[np.newaxis]).cuda(), torch.FloatTensor(obj_triangles[np.newaxis]).cuda()
            )
            percentage_seen = (1-exterior).sum().float()/len(points_object)
            percentage_seen = percentage_seen.cpu().data.numpy()
            percentages.append(percentage_seen)

        self.percentage_visible_points = np.array(percentages)



# GET ANGLE BETWEEN CAMERA AND MAXIMUM DIRECTION OF THE OBJECT:
    def get_angle_camera_maximumdirection(self):
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        angles = []
        for index in range(self._dataset_size):
            points_object, faces_object = self.load_obj_details(index)
            pca.fit(points_object)

            normal = pca.components_[0]
            cam2obj = np.mean(points_object, 0)
            cam2obj = cam2obj/np.sqrt((cam2obj**2).sum())
            num = np.dot(cam2obj, normal)
            angle = np.arccos(num)
            angle = min(angle, np.pi - angle)
            angles.append(angle)

        self.angles = np.array(angles)
