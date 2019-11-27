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

from sklearn.decomposition import PCA
import point_cloud_utils as pcu
import pandas as pd
import cv2

def fast_load_obj(file_obj, **kwargs):
    """
    Code slightly adapted from trimesh (https://github.com/mikedh/trimesh) 
    Thanks to Michael Dawson-Haggerty for this great library !
    loads an ascii wavefront obj file_obj into kwargs
    for the trimesh constructor.

    vertices with the same position but different normals or uvs
    are split into multiple vertices.

    colors are discarded.

    parameters
    ----------
    file_obj : file object
                   containing a wavefront file

    returns
    ----------
    loaded : dict
                kwargs for trimesh constructor
    """

    # make sure text is utf-8 with only \n newlines
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.replace('\r\n', '\n').replace('\r', '\n') + ' \n'

    meshes = []
    def append_mesh():
        # append kwargs for a trimesh constructor
        # to our list of meshes
        if len(current['f']) > 0:
            # get vertices as clean numpy array
            vertices = np.array(
                current['v'], dtype=np.float64).reshape((-1, 3))
            # do the same for faces
            faces = np.array(current['f'], dtype=np.int64).reshape((-1, 3))

            # get keys and values of remap as numpy arrays
            # we are going to try to preserve the order as
            # much as possible by sorting by remap key
            keys, values = (np.array(list(remap.keys())),
                            np.array(list(remap.values())))
            # new order of vertices
            vert_order = values[keys.argsort()]
            # we need to mask to preserve index relationship
            # between faces and vertices
            face_order = np.zeros(len(vertices), dtype=np.int64)
            face_order[vert_order] = np.arange(len(vertices), dtype=np.int64)

            # apply the ordering and put into kwarg dict
            loaded = {
                'vertices': vertices[vert_order],
                'faces': face_order[faces],
                'metadata': {}
            }

            # build face groups information
            # faces didn't move around so we don't have to reindex
            if len(current['g']) > 0:
                face_groups = np.zeros(len(current['f']) // 3, dtype=np.int64)
                for idx, start_f in current['g']:
                    face_groups[start_f:] = idx
                loaded['metadata']['face_groups'] = face_groups

            # we're done, append the loaded mesh kwarg dict
            meshes.append(loaded)

    attribs = {k: [] for k in ['v']}
    current = {k: [] for k in ['v', 'f', 'g']}
    # remap vertex indexes {str key: int index}
    remap = {}
    next_idx = 0
    group_idx = 0

    for line in text.split("\n"):
        line_split = line.strip().split()
        if len(line_split) < 2:
            continue
        if line_split[0] in attribs:
            # v, vt, or vn
            # vertex, vertex texture, or vertex normal
            # only parse 3 values, ignore colors
            attribs[line_split[0]].append([float(x) for x in line_split[1:4]])
        elif line_split[0] == 'f':
            # a face
            ft = line_split[1:]
            if len(ft) == 4:
                # hasty triangulation of quad
                ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
            for f in ft:
                # loop through each vertex reference of a face
                # we are reshaping later into (n,3)
                if f not in remap:
                    remap[f] = next_idx
                    next_idx += 1
                    # faces are "vertex index"/"vertex texture"/"vertex normal"
                    # you are allowed to leave a value blank, which .split
                    # will handle by nicely maintaining the index
                    f_split = f.split('/')
                    current['v'].append(attribs['v'][int(f_split[0]) - 1])
                current['f'].append(remap[f])
        elif line_split[0] == 'o':
            # defining a new object
            append_mesh()
            # reset current to empty lists
            current = {k: [] for k in current.keys()}
            remap = {}
            next_idx = 0
            group_idx = 0

        elif line_split[0] == 'g':
            # defining a new group
            group_idx += 1
            current['g'].append((group_idx, len(current['f']) // 3))

    if next_idx > 0:
        append_mesh()

    return meshes


def getBB(depth):
    objs = ndimage.find_objects((depth > 0)*1)[0]
    x, x2 = objs[0].start, objs[0].stop
    y, y2 = objs[1].start, objs[1].stop

    return x, y, x2, y2

cam_intr = np.array([[480., 0., 128.], [0., 480., 128.],
                          [0., 0., 1.]]).astype(np.float32)
cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.],
                          [0., 0., -1., 0.]]).astype(np.float32)
fov = 2*np.arctan(256/(2*480))

def point_cloud(depth):
    cx = 320
    cy = 240
    fx = 640
    fy = 480

#    cx = cy = 128
#    fx = fy = 480
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    valid = (depth > 0)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)
    return x[valid], y[valid]*-1, z[valid]*-1

class Dataset(DatasetBase):
    def __init__(self, opt, mode):
        super(Dataset, self).__init__(opt, mode)
        self._name = 'Dataset_obman'
        self.pc_name = socket.gethostname()

        self.obj = self._opt.data_dir
        #self.obj = 'coffee_mug'

        # read dataset
        self._read_dataset_paths()
        self.noise_units = 12

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        img = np.float32(self._depth_imgs[index])
        #img = img - self.means_depth
        img = img / self.std_depth
        #print("Further image is not normalize! JUST CORRECT THIS IF WE TRAIN THIS MODEL AGAIN!!!!")

        self.noise_units = 12
        #noise = np.random.normal(0, 1.0, (self._opt.image_size, self._opt.image_size))
        noise = np.random.normal(0, 1.0, (self.noise_units, 1, 1))
        noise = noise.repeat(128, 1).repeat(128, 2)

        # TODO: CONSIDER LOADING ALL THESE AND SAMPLE BEFOREHAND!
        points_object = 0
        faces_object = 0

        obj_verts = self.model_verts/1000 - self.obj_translations[index]
        obj_faces = self.model_faces

        # pack data
        sample = {'rgb_img': self._rgb_imgs[index],
                  'realsizedepth_img': self._realsizedepth_imgs[index],
                  'depth_img': img,
                  'further_depth_img': img,
                  'coords_3d': 0,
                  'noise_img': noise,
                  'label': 0,
                  'cluster': 0,
                  'verts_3d': 0,
                  'coords_3d': 0,
                  'pca_poses': 0,
                  '3d_points_object_resampled': 0,
                  '3d_points_object':obj_verts,
                  '3d_faces_object':obj_faces,
                  'object_directions': self.obj_normals[index],
                  'object_translation': self.obj_translations[index],
                  'mano_rotation': 0,
                  'mano_translation': 0,
                  }

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):
        # read ids
        self._ids = self._read_ids()
        self._ids.sort()

        # DEMO:
        self._ids = self._ids[:50]
        #self._ids = self._ids[:200]

        # dataset size
        self._dataset_size = len(self._ids)

        # Read cad model
        model_path='/tmp-network/fast/ecorona/data/objects_cadmodels/'+str(self.obj)+'.obj'
        with open(model_path, 'r') as m_f:
            mesh = fast_load_obj(m_f)[0]

        self.model_verts = mesh['vertices']
        self.model_faces = mesh['faces']

        # read imgs
        self._realsizedepth_imgs, self._depth_imgs, self._rgb_imgs = self._read_imgs()

        ## Center imgs
        #for i in range(len(self._depth_imgs)):
        #    x, y, x2, y2 = getBB(self._depth_imgs[i])
        #    obj = self._depth_imgs[i][x:x2, y:y2].copy()
        #    self._depth_imgs[i] = 0
        #    new_x = int(self._opt.image_size/2 - (x2-x)/2)
        #    new_y = int(self._opt.image_size/2 - (y2-y)/2)
        #    self._depth_imgs[i][new_x:new_x+(x2-x), new_y:new_y + (y2-y)] = obj

        #for i in range(len(self._further_depth_imgs)):
        #    x, y, x2, y2 = getBB(self._further_depth_imgs[i])
        #    obj = self._further_depth_imgs[i][x:x2, y:y2].copy()
        #    self._further_depth_imgs[i] = 0
        #    new_x = int(self._opt.image_size/2 - (x2-x)/2)
        #    new_y = int(self._opt.image_size/2 - (y2-y)/2)
        #    self._further_depth_imgs[i][new_x:new_x+(x2-x), new_y:new_y + (y2-y)] = obj

        pca = PCA(n_components=2)

        self.obj_normals = []
        self.obj_translations = []
        for i in range(len(self._depth_imgs)):
            x, y, z = point_cloud(self._realsizedepth_imgs[0])
            reconstruction_points = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), 1)
            try:
                pca.fit(np.transpose([x, y, z]))
            except:
                print("Failed for object " + str(i))
                from IPython import embed
                embed()

            self.obj_normals.append(pca.components_)
            self.obj_translations.append([x.mean(), y.mean(), z.mean()])

        # read obj characteristics
        #self.obj_normals = np.load(self._root_dir + '/obtained_normals_objects.npy')
        #self.obj_translations = np.load(self._root_dir + '/obtained_translations_objects.npy')

        # Obtained from train set
        self.means_rgb = 112.61943782666667
        self.std_rgb = 71.08420034477948
        self.means_depth = 0.04484606927111739
        self.std_depth = 0.16042266926713555

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


    def _read_ids(self):
        import glob
        files = glob.glob('/tmp-network/fast/ecorona/data/rgbd-objects/'+str(self.obj)+'/'+str(self.obj)+'_1/*depth.png')
        files = [f.replace('_depth.png', '') for f in files]
        return files

    def _read_imgs(self):
        data = []
        realsizedata = []
        rgbs = []
        for i in range(self._dataset_size):
            filepath = self._ids[i]
            rgb = cv2.imread(filepath+'.png')[:, :, ::-1]
            img = cv2.imread(filepath+'_depth.png', 2).astype(np.float32)
            mask = mask = cv2.imread(filepath+'_mask.png', 2)
            rgb[mask==0] = 0
            img[mask==0] = 0
            img = img/1000

            ##position = pd.read_csv(filepath+'_loc.txt', header=None).values[0]
            #img[:position[1], :] = 0
            #img[position[1]+100:, :] = 0
            #img[:, :position[0]] = 0
            #img[:, position[0]+100:] = 0
            #depth = img[position[1]+50, position[0]+50]
            ##img[img>depth+0.1] = 0

            realsizedata.append(img)
            img = resize(img[20:-20, 20:-20] - 1, (self._opt.image_size, self._opt.image_size))
            data.append(np.float16(img+1))
            rgbs.append(rgb)

        return np.array(realsizedata), np.array(data), np.array(rgbs)

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

