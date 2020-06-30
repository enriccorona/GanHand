import os
import argparse
import glob
import cv2
from utils import cv_utils
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import numpy as np
from models.models import ModelsFactory
from options.test_options import TestOptions

import time
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from utils.tb_visualizer import TBVisualizer
from collections import OrderedDict
import os

#import pyhull.convex_hull as cvh
import cvxopt as cvx

from manopth import rodrigues_layer
from manopth.tensutils import th_posemap_axisang, make_list, th_pack, subtract_flat_id, th_with_zeros

import trimesh
import pyquaternion
import utils.plots as plot_utils
from utils import ycb_utils

from matplotlib import pyplot as plt
import point_cloud_utils as pcu

import tqdm

from utils import obman_utils as obman
from scipy.stats import pearsonr

from utils import contactutils


def _get_touching_distances(hand_points, object_points):
    if len(object_points) > 60000:
        object_points = object_points[:60000]

    distances = []
    n1 = len(hand_points[0])
    # TODO: HAVE TO DO IT IN A LOOP SINCE OBJECTS ALL HAVE DIFFERENT AMOUNT OF VERTICES
    for i in range(len(hand_points)):
        n2 = len(object_points)

        matrix1 = hand_points[i].unsqueeze(0).repeat(n2, 1, 1)
        if torch.cuda.is_available():
            matrix2 = torch.FloatTensor(object_points).cuda().unsqueeze(1).repeat(1, n1, 1)
        else:
            matrix2 = torch.FloatTensor(object_points).unsqueeze(1).repeat(1, n1, 1)
        dists = torch.sqrt(((matrix1-matrix2)**2).sum(-1))
        distances.append(dists.min(0)[0])

    return torch.stack(distances)

#@staticmethod
def min_norm_vector_in_facet(facet, wrench_regularizer=1e-10):
    """ Finds the minimum norm point in the convex hull of a given facet (aka simplex) by solving a QP.

    Parameters
    ----------
    facet : 6xN :obj:`numpy.ndarray`
        vectors forming the facet
    wrench_regularizer : float
        small float to make quadratic program positive semidefinite

    Returns
    -------
    float
        minimum norm of any point in the convex hull of the facet
    Nx1 :obj:`numpy.ndarray`
        vector of coefficients that achieves the minimum
    """
    dim = facet.shape[1] # num vertices in facet

    # create alpha weights for vertices of facet
    G = facet.T.dot(facet)
    grasp_matrix = G + wrench_regularizer * np.eye(G.shape[0])

    # Solve QP to minimize .5 x'Px + q'x subject to Gx <= h, Ax = b
    P = cvx.matrix(2 * grasp_matrix)   # quadratic cost for Euclidean dist
    q = cvx.matrix(np.zeros((dim, 1)))
    G = cvx.matrix(-np.eye(dim))       # greater than zero constraint
    h = cvx.matrix(np.zeros((dim, 1)))
    A = cvx.matrix(np.ones((1, dim)))  # sum constraint to enforce convex
    b = cvx.matrix(np.ones(1))         # combinations of vertices

    sol = cvx.solvers.qp(P, q, G, h, A, b)
    v = np.array(sol['x'])
    min_norm = np.sqrt(sol['primal objective'])

    return abs(min_norm), v

#def th_with_zeros(tensor):
#    batch_size = tensor.shape[0]
#    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
#    padding.requires_grad = False
#
#    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
#    cat_res = torch.cat(concat_list, 1)
#    return cat_res


def grasp_matrix(forces, torques, normals, soft_fingers=False,
                 finger_radius=0.005, params=None):
    if params is not None and 'finger_radius' in params.keys():
        finger_radius = params.finger_radius
    num_forces = forces.shape[1]
    num_torques = torques.shape[1]
    if num_forces != num_torques:
        raise ValueError('Need same number of forces and torques')

    num_cols = num_forces
    if soft_fingers:
        num_normals = 2
        if normals.ndim > 1:
            num_normals = 2*normals.shape[1]
        num_cols = num_cols + num_normals

    torque_scaling = 1
    G = np.zeros([6, num_cols])
    for i in range(num_forces):
        G[:3,i] = forces[:,i]
        #G[3:,i] = forces[:,i] # ZEROS
        G[3:,i] = torque_scaling * torques[:,i]

    if soft_fingers:
        torsion = np.pi * finger_radius**2 * params.friction_coef * normals * params.torque_scaling
        pos_normal_i = -num_normals
        neg_normal_i = -num_normals + num_normals / 2
        G[3:,pos_normal_i:neg_normal_i] = torsion
        G[3:,neg_normal_i:] = -torsion

    return G

def get_normal_face(p1, p2, p3):
    U = p2 - p1
    V = p3 - p1
    Nx = U[1]*V[2] - U[2]*V[1]
    Ny = U[2]*V[0] - U[0]*V[2]
    Nz = U[0]*V[1] - U[1]*V[0]
    return [Nx, Ny, Nz]

def get_distance_vertices(obj, hand):
    distances = []
    n1 = len(hand)
    n2 = len(obj)

    matrix1 = hand[np.newaxis].repeat(n2, 0)
    matrix2 = obj[:, np.newaxis].repeat(n1, 1)
    dists = np.sqrt(((matrix1-matrix2)**2).sum(-1))
    return dists.min(0)


class Test:
    def __init__(self):

        # TO GET THEM:
        # clusters_pose_map, clusters_rot_map, clusters_root_rot = self.get_rot_map(self._model.clusters_tensor, torch.zeros((25, 3)).cuda())
        #for i in range(25):
        #    import matplotlib.pyplot
        #    from mpl_toolkits.mplot3d import Axes3D
        #    ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')
        #    #i = 0
        #    add_group_meshs(ax, cluster_verts[i].cpu().data.numpy(), hand_faces, c='b')
        #    cam_equal_aspect_3d(ax, cluster_verts[i].cpu().data.numpy())
        #    print(i)
        #    matplotlib.pyplot.pause(1)
        #    matplotlib.pyplot.close()

        # FINGER LIMIT ANGLE:
        #self.limit_bigfinger = torch.FloatTensor([1.0222, 0.0996, 0.7302]) # 36:39
        #self.limit_bigfinger = torch.FloatTensor([1.2030, 0.12, 0.25]) # 36:39
        #self.limit_bigfinger = torch.FloatTensor([1.2, -0.4, 0.25]) # 36:39
        self.limit_bigfinger = torch.FloatTensor([1.2, -0.6, 0.25]) # 36:39
        self.limit_index = torch.FloatTensor([-0.0827, -0.4389,  1.5193]) # 0:3
        self.limit_middlefinger = torch.FloatTensor([-2.9802e-08, -7.4506e-09,  1.4932e+00]) # 9:12
        self.limit_fourth = torch.FloatTensor([0.1505, 0.3769, 1.5090]) # 27:30
        self.limit_small = torch.FloatTensor([-0.6235,  0.0275,  1.0519]) # 18:21
        if torch.cuda.is_available():
            self.limit_bigfinger = self.limit_bigfinger.cuda()
            self.limit_index = self.limit_index.cuda()
            self.limit_middlefinger = self.limit_middlefinger.cuda()
            self.limit_fourth = self.limit_fourth.cuda()
            self.limit_small = self.limit_small.cuda()



        self._bigfinger_vertices = [697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724,
        725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737,
        738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750,
        751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763,
        764, 765, 766, 767, 768]

        self._indexfinger_vertices = [46, 47, 48, 49, 56, 57, 58, 59, 86, 87, 133, 134, 155, 156, 164, 165, 166, 167, 174, 175, 189, 194, 195, 212, 213, 221, 222, 223, 224, 225, 226, 237, 238, 272, 273, 280, 281, 282, 283, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355]

        self._middlefinger_vertices = [356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367,
        372, 373, 374, 375, 376, 377, 381,
        382, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394,
        395, 396, 397, 398, 400, 401, 402, 403, 404, 405, 406, 407,
        408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420,
        421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433,
        434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446,
        447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459,
        460, 461, 462, 463, 464, 465, 466, 467]

        self._fourthfinger_vertices = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 482, 483, 484, 485, 486, 487, 491, 492,
        495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507,
        508, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520,
        521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533,
        534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
        547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
        560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
        573, 574, 575, 576, 577, 578]

        self._smallfinger_vertices = [580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591,
        598, 599, 600, 601, 602, 603,
        609, 610, 613, 614, 615, 616, 617, 618, 619, 620,
        621, 622, 623, 624, 625, 626, 628, 629, 630, 631, 632, 633,
        634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646,
        647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659,
        660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672,
        673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685,
        686, 687, 688, 689, 690, 691, 692, 693, 694, 695]


        self._opt = TestOptions().parse()
        #assert self._opt.load_epoch > 0, 'Use command --load_epoch to indicate the epoch you want to load - and choose a trained model'

        # Let's set batch size at 2 since we're only getting one image so far
        self._opt.batch_size = 1

        #print("Loading validation set instead of test set for faster debugging!!")
        #print("Loading validation set instead of test set for faster debugging!!")
        #print("Loading validation set instead of test set for faster debugging!!")
        #print("Loading validation set instead of test set for faster debugging!!")
        #print("Loading validation set instead of test set for faster debugging!!")

        self._opt.n_threads_train = self._opt.n_threads_test
        #data_loader_test = CustomDatasetDataLoader(self._opt, mode='train')
        #data_loader_test = CustomDatasetDataLoader(self._opt, mode='val')
        data_loader_test = CustomDatasetDataLoader(self._opt, mode='test')
        self._dataset_test = data_loader_test.load_data()
        self._dataset_test_size = len(data_loader_test)
        print('#test images = %d' % self._dataset_test_size)

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._tb_visualizer = TBVisualizer(self._opt)

        self._total_steps = self._dataset_test_size
        self._display_visualizer_test(20, self._total_steps)

    def _display_visualizer_test(self, i_epoch, total_steps):
        test_start_time = time.time()
        threshold_visuals = 200
        weight_qualities = 70
        weight_collisions = 0.1

        # set model to eval
        self._model.set_eval()

        # evaluate self._opt.num_iters_validate epochs
        test_errors = OrderedDict()
        iters = 0
        hand_faces = self._model._MANO.th_faces.cpu().data.numpy()

        from sklearn.cluster import MiniBatchKMeans
        sampled_rotations = np.load('sample_rotations.npy', allow_pickle=True)
        #howmanysamples = 4
        howmanysamples = 49

        threshold_intersections = 100 #150

        print("%d TO BE PROCESSED SAMPLES"%(len(self._dataset_test)))
        print("SAMPLING %d POSSIBLE GRASPS"%(howmanysamples))
        print("IF PROCESS TAKES TOO MUCH TIME; REDUCE VALUE OF howmanysamples OR TEST WITH SCENES THAT HAVE LESS AMOUNT OF OBJECTS")
        print("FOR FASTER INFERENCE TIME, CHANGE GRASP METRIC")
        print("MAKING VIDEOS TAKES A LONG TIME, GET JUST PREDICTION FRAMES FOR FASTER INFERENCE")

        for i_test_batch, test_batch in enumerate(self._dataset_test):
            print("PROCESSING TEST SAMPLE %d"%(i_test_batch))
            import utils.plots as plot_utils
            self._model.set_input(test_batch)

            numobjects = len(self._model._input_obj_verts[0])
            all_faces = np.zeros((0, 3))
            all_verts = np.zeros((0, 3))

            for j in range(numobjects):
                all_faces = np.concatenate((all_faces, self._model._input_obj_faces[0][j] + len(all_verts)))
                all_verts = np.concatenate((all_verts, self._model._input_obj_verts[0][j]))

            numobjects = len(test_batch['mask_img'][0])
            hand_verts = []
            hand_Ts = []
            hand_Rs = []
            hand_HRs = []
            for j in range(numobjects):
                sample_hand_verts = []
                sample_hand_T = []
                sample_hand_HR = []
                sample_hand_R = []
                sample_grasp_stability = []
                obj_verts = self._model._input_obj_verts[0][j]
                obj_faces = self._model._input_obj_faces[0][j]
                resampled_obj_verts = self._model._input_obj_resampled_verts[0][j]
                for k in range(howmanysamples):
                    self._model._use_approach = True
                    self._model._approach_orientation = sampled_rotations[howmanysamples-2][k]

                    self._model._input_mask_img = test_batch['mask_img'][0][j].float().unsqueeze(0).unsqueeze(0).cuda()
                    self._model._input_object_id = [j]
                    self._model._input_center_objects = torch.FloatTensor([self._model._input_obj_resampled_verts[0][j].mean(0)]).cuda()
                    self._model.forward(keep_data_for_visuals=True)

                    verts = self._model._refined_handpose

                    forces_post, torques_post, normals_post, finger_is_touching = self.get_contact_points(verts, obj_verts, self._model._MANO.th_faces.cpu().data.numpy(), obj_faces, resampled_obj_verts)

                    if len(forces_post) < 3 or not finger_is_touching[-1]:
                        continue # Not enough contacts
                    interpenetration = self.get_interpenetration(verts, test_batch['3d_points_object'][0], test_batch['3d_faces_object'][0])

                    if interpenetration > threshold_intersections:
                        continue # Not available due to intersections

                    sample_hand_verts.append(verts)
                    sample_hand_T.append(self._model._refined_T.cpu().data.numpy())
                    sample_hand_HR.append(self._model._refined_HR.cpu().data.numpy())
                    sample_hand_R.append(self._model._refined_R.cpu().data.numpy())

                    # Grasp metric:
                    # NOTE: FOR FASTER RESULT, USE METRIC OTHER THAN THIS, WHICH TAKES A LONG TIME
                    G = grasp_matrix(np.array(forces_post).transpose(), np.array(torques_post).transpose(), np.array(normals_post).transpose())
                    grasp_metric = min_norm_vector_in_facet(G)[0]
                    # Simulation displacement
                    #distance = obman.run_simulation(verts, self._model._MANO.th_faces.cpu().data.numpy(), obj_verts.copy(), obj_faces)
                    #sample_grasp_stability.append(distance)
                    sample_grasp_stability.append(grasp_metric)

                if len(sample_hand_verts) == 0:
                    continue

                most_stable_grasp = np.argmax(sample_grasp_stability)
                hand_verts.append(sample_hand_verts[most_stable_grasp])
                hand_Ts.append(sample_hand_T[most_stable_grasp])
                hand_HRs.append(sample_hand_HR[most_stable_grasp])
                hand_Rs.append(sample_hand_R[most_stable_grasp])

            num_frames_video = 50
            previous_verts = np.zeros((0, 3))
            previous_faces = np.zeros((0, 3))
            video_verts = []
            video_faces = []
            for j in range(len(hand_Ts)):
                T = torch.FloatTensor(hand_Ts[j]).cuda()
                R = torch.FloatTensor(hand_Rs[j]).cuda()
                HR = torch.FloatTensor(hand_HRs[j]).cuda()
                for i in range(num_frames_video + 1):
                    interpol_HR = torch.zeros((1, 45)).cuda()*(num_frames_video - i)/num_frames_video + HR*(i)/num_frames_video

                    points_3d, _ = self._model._MANO(torch.cat((R, interpol_HR), -1), th_trans=T)
                    points_3d = points_3d[0].cpu().data.numpy()/1000
                    video_faces.append(np.concatenate((previous_faces.copy(), hand_faces + len(previous_verts))))
                    video_verts.append(np.concatenate((previous_verts.copy(), points_3d.copy())))

                previous_faces = np.concatenate((previous_faces.copy(), hand_faces + len(previous_verts)))
                previous_verts = np.concatenate((previous_verts.copy(), points_3d.copy()))

            print("RENDERING VIDEO")
            frames_w_objects = []
            frames_wo_objects = []
            for frame in tqdm.tqdm(range(len(video_verts))):
                img = plot_utils.render_hand_objects_on_img(np.float32(video_verts[frame]), np.int32(video_faces[frame]), np.float32(all_verts), np.int32(all_faces), self._model._input_fullsize_img[0], self._model._input_cam_intrinsics[0])
                #img_2 = plot_utils.render_occluded_hand_on_img(np.float32(video_verts[frame]), np.int32(video_faces[frame]), np.float32(all_verts), np.int32(all_faces), self._model._input_fullsize_img[0], self._model._input_cam_intrinsics[0])
                frames_w_objects.append(img[:,:,::-1])
                #frames_wo_objects.append(img_2[:,:,::-1])

            plot_utils.save_video(frames_w_objects, 'results/video_%d_w_objs'%(i_test_batch))
            #plot_utils.save_video(frames_wo_objects, 'results/video_%d_wo_objs'%(i_test_batch))

    def get_interpenetration(self, verts, obj_verts, obj_faces):
        verts = torch.FloatTensor(verts).cuda()

        numobjects = len(obj_verts)
        all_triangles = []
        all_verts = []
        for j in range(numobjects):
            obj_triangles = obj_verts[j][obj_faces[j]]
            obj_triangles = torch.FloatTensor(obj_triangles).cuda()
            all_triangles.append(obj_triangles)
            all_verts.append(torch.FloatTensor(obj_verts[j]).cuda())
        all_triangles = torch.cat(all_triangles)
        all_verts = torch.cat(all_verts)

        exterior = contactutils.batch_mesh_contains_points(
            verts.unsqueeze(0), all_triangles.unsqueeze(0)
        )
        return (~exterior).sum().item()

    def get_contact_points(self, hand, obj, hand_faces, obj_faces, resampled_obj_verts):
        finger_vertices = [self._indexfinger_vertices, self._middlefinger_vertices, self._fourthfinger_vertices, self._smallfinger_vertices, self._bigfinger_vertices]
        #finger_vertices = [[309, 317, 318, 319, 320, 322, 323, 324, 325,
        #   326, 327, 328, 329, 332, 333, 337, 338, 339, 343, 347, 348, 349,
        #   350, 351, 352, 353, 354, 355], [429, 433, 434, 435, 436, 437, 438,
        #   439, 442, 443, 444, 455, 461, 462, 463, 465, 466, 467], [547, 548,
        #   549, 550, 553, 566, 573, 578], [657, 661, 662, 664, 665, 666, 667,
        #   670, 671, 672, 677, 678, 683, 686, 687, 688, 689, 690, 691, 692,
        #   693, 694, 695], [736, 737, 738, 739, 740, 741, 743, 753, 754, 755,
        #   756, 757, 759, 760, 761, 762, 763, 764, 766, 767, 768],
        #   [73,  96,  98,  99, 772, 774, 775, 777]]

        forces = []
        torques = []
        normals = []
        finger_is_touching = np.zeros(5)
        #resampled_obj_verts = pcu.sample_mesh_lloyd(obj.astype(np.float32), obj_faces.astype(np.int32), 2000)
        threshold = 0.004 # m
        for i in range(len(finger_vertices)):
            dists = get_distance_vertices(resampled_obj_verts, hand[finger_vertices[i]])
            if np.min(dists) < threshold:
                finger_is_touching[i] = 1
                faces = np.where(finger_vertices[i][np.argmin(dists)] == hand_faces)[0]
                normal = []
                for j in range(len(faces)):
                    normal.append(get_normal_face(hand[hand_faces[faces[j], 0]], hand[hand_faces[faces[j], 1]], hand[hand_faces[faces[j], 2]]))
                normal = np.mean(normal, 0) * 1e5 # Multiply by large number to avoid **2 going to zero
                normal = normal/np.sqrt((np.array(normal)**2).sum())
                torques.append([0, 0, 0])
                normals.append(normal)
                forces.append(normal)

        return forces, torques, normals, finger_is_touching

if __name__ == '__main__':
    Test()
