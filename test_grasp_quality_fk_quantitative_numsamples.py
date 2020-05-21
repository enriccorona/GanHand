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

import pyhull.convex_hull as cvh
import cvxopt as cvx

from manopth import rodrigues_layer
from manopth.tensutils import th_posemap_axisang, make_list, th_pack, subtract_flat_id, th_with_zeros

import trimesh
import pyquaternion
import utils.plots as plot_utils

from matplotlib import pyplot as plt
import point_cloud_utils as pcu

import tqdm

from utils import obman_utils as obman
from scipy.stats import pearsonr


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

        num_samples = self._opt.num_samples
        #num_samples = 1
        #num_samples = 3
        #num_samples = 5
        #num_samples = 8
        #num_samples = 10
        #num_samples = 12
        #num_samples = 15
        #num_samples = 20
        #num_samples = 25 
        #num_samples = 50
        #save_viz=True
        save_viz=False
        hand_faces = self._model._MANO.th_faces.cpu().data.numpy()

        print("OPTIMIZING %d"%(num_samples))

        sampled_rotations = np.load('sample_rotations.npy', allow_pickle=True)


        results_times = []

        results_graspit_post_accgraspit = []
        results_contacts_post_accgraspit = []
        results_simulation_post_accgraspit = []
        results_intersection_post_accgraspit = []

        results_graspit_post_accnumcontacts = []
        results_contacts_post_accnumcontacts = []
        results_simulation_post_accnumcontacts = []
        results_intersection_post_accnumcontacts = []

        results_graspit_post_accinterpenetration = []
        results_contacts_post_accinterpenetration = []
        results_simulation_post_accinterpenetration = []
        results_intersection_post_accinterpenetration = []

        results_times = []

        # Analysis:
        results_angles = []
        results_percentage_visible_points = []

        thresh_collisions = 250 # realistic

        for i_test_batch, test_batch in enumerate(self._dataset_test):
            #if i_test_batch %5 != 0:
            #    continue

            print("i_test_batch=%d"%(i_test_batch))
            print("i_test_batch=%d"%(i_test_batch))
            print("i_test_batch=%d"%(i_test_batch))
            print("i_test_batch=%d"%(i_test_batch))
            print("i_test_batch=%d"%(i_test_batch))
            print("")

            batchsize = len(test_batch['label'])
            refined_visuals = []
            visuals = []
            qualities = []

            times = []
            graspit_measure_post = []
            number_contacts_post = []
            simul_distances_post = []
            intersections_post = []
            maximum_number_contacts = -1
            minimum_number_collisions = 2000

            self._model.set_input(test_batch)

            verts_post = []
            min_collisions_post = 999999

            #percentage_visible_points = test_batch['percentage_visible_points']
            #angle_object_camera = test_batch['angle_between_normal_and_cameradirection']

            for j in tqdm.tqdm(range(num_samples)):
                start = time.time()
                self._model._use_approach = True
                self._model._approach_orientation = sampled_rotations[num_samples-2][j]

                self._model.forward(keep_data_for_visuals=True)
                times.append(time.time()-start)

                hand_vertices = self._model._refined_handpose

                obj_verts = self._model._input_obj_verts[0]
                obj_faces = self._model._input_obj_faces[0]

                # AFTER OPTIMIZATION:
                forces_post, torques_post, normals_post = self.get_contact_points(hand_vertices, obj_verts, hand_faces, obj_faces, obj_verts)
                num_collisions_post = self.get_intersections_allhand(hand_vertices, obj_verts, obj_faces)

                if num_collisions_post < min_collisions_post or num_collisions_post < thresh_collisions:
                    min_collisions_post = num_collisions_post

                    number_contacts_post.append(len(forces_post))
                    if len(forces_post) > 2:
                        G = grasp_matrix(np.array(forces_post).transpose(), np.array(torques_post).transpose(), np.array(normals_post).transpose())
                        metric = min_norm_vector_in_facet(G)[0]
                        graspit_measure_post.append(metric)
                    else:
                        graspit_measure_post.append(0.0)

                    distance = obman.run_simulation(hand_vertices, hand_faces, obj_verts.copy(), obj_faces)
                    intersection = obman.get_intersection(hand_vertices, hand_faces, obj_verts.copy(), obj_faces)
                    simul_distances_post.append(distance)
                    intersections_post.append(intersection)
                else:
                    number_contacts_post.append(0)
                    graspit_measure_post.append(0.0)
                    simul_distances_post.append(0.15)
                    intersections_post.append(5e-5)

                if save_viz:
                    verts_post.append(hand_vertices)

            graspit_measure_post = np.array(graspit_measure_post)
            number_contacts_post = np.array(number_contacts_post)
            simul_distances_post = np.array(simul_distances_post)
            intersections_post = np.array(intersections_post)
            times = np.array(times)

            print(number_contacts_post)

            index_best_grasp = graspit_measure_post.argmax()

            results_graspit_post_accgraspit.append(graspit_measure_post[index_best_grasp])
            results_contacts_post_accgraspit.append(number_contacts_post[index_best_grasp])
            results_simulation_post_accgraspit.append(simul_distances_post[index_best_grasp])
            results_intersection_post_accgraspit.append(intersections_post[index_best_grasp])

            index_best_grasp = number_contacts_post.argmax()

            results_graspit_post_accnumcontacts.append(graspit_measure_post[index_best_grasp])
            results_contacts_post_accnumcontacts.append(number_contacts_post[index_best_grasp])
            results_simulation_post_accnumcontacts.append(simul_distances_post[index_best_grasp])
            results_intersection_post_accnumcontacts.append(intersections_post[index_best_grasp])

            index_best_grasp = intersections_post.argmin()

            results_graspit_post_accinterpenetration.append(graspit_measure_post[index_best_grasp])
            results_contacts_post_accinterpenetration.append(number_contacts_post[index_best_grasp])
            results_simulation_post_accinterpenetration.append(simul_distances_post[index_best_grasp])
            results_intersection_post_accinterpenetration.append(intersections_post[index_best_grasp])


            results_times.append(times.sum())

        from IPython import embed
        embed()

        #results_angles = np.array(results_angles)[:, 0]
        #results_percentage_visible_points = np.array(results_percentage_visible_points)[:, 0]

        print("END")

        print("")
        print("")
        print("BEST ACCORDING TO GRASPIT METRIC:")

        print("GRASPIT")
        print(np.mean(results_graspit_post_accgraspit))
        print("CONTACTS")
        print(np.mean(results_contacts_post_accgraspit))
        print("SIMULATION")
        print(np.mean(results_simulation_post_accgraspit))
        print("INTERSECTIONS")
        print(np.mean(results_intersection_post_accgraspit))

        print("")
        print("")
        print("BEST ACCORDING TO NUMBER OF CONTACTS METRIC:")

        print("GRASPIT")
        print(np.mean(results_graspit_post_accnumcontacts))
        print("CONTACTS")
        print(np.mean(results_contacts_post_accnumcontacts))
        print("SIMULATION")
        print(np.mean(results_simulation_post_accnumcontacts))
        print("INTERSECTIONS")
        print(np.mean(results_intersection_post_accnumcontacts))

        print("")
        print("")
        print("BEST ACCORDING TO LEAST INTERPENETRATION:")
        
        print("GRASPIT")
        print(np.mean(results_graspit_post_accinterpenetration))
        print("CONTACTS")
        print(np.mean(results_contacts_post_accinterpenetration))
        print("SIMULATION")
        print(np.mean(results_simulation_post_accinterpenetration))
        print("INTERSECTIONS")
        print(np.mean(results_intersection_post_accinterpenetration))

        print("")
        print("")
        print("TIMES")
        print(np.mean(results_times))
        

    def get_joint_Hs(self, HR, R, T):
        th_full_hand_pose = HR.unsqueeze(0).mm(self._model._MANO.th_selected_comps)
        th_full_pose = torch.cat([
            R.unsqueeze(0), self._model._MANO.th_hands_mean + th_full_hand_pose
            ], 1)
        th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
        th_full_pose = th_full_pose.view(1, -1, 3)
        root_rot = rodrigues_layer.batch_rodrigues(
            th_full_pose[:, 0]).view(1, 3, 3)

        th_v_shaped = torch.matmul(self._model._MANO.th_shapedirs,
                                   self._model._MANO.th_betas.transpose(1, 0)).permute(
                                       2, 0, 1) + self._model._MANO.th_v_template
        th_j = torch.matmul(self._model._MANO.th_J_regressor, th_v_shaped).repeat(
            1, 1, 1)

        root_j = th_j[:, 0, :].contiguous().view(1, 3, 1)
        th_results = []
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))
        angle_parents = [4294967295, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
        # Rotate each part
        for i in range(15):
            i_val = int(i + 1)
            joint_rot = th_rot_map[:, (i_val - 1) * 9:i_val *
                                   9].contiguous().view(1, 3, 3)
            joint_j = th_j[:, i_val, :].contiguous().view(1, 3, 1)
            parent = make_list(angle_parents)[i_val]
            parent_j = th_j[:, parent, :].contiguous().view(1, 3, 1)
            joint_rel_transform = th_with_zeros(
                torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(
                torch.matmul(th_results[parent], joint_rel_transform))

        Hs = torch.cat(th_results)
        Hs[:, :3, 3] = Hs[:, :3, 3] + T
        return Hs


    def get_Hs_fingers(self, HR, R, T):
        th_full_hand_pose = HR.unsqueeze(0).mm(self._model._MANO.th_selected_comps)
        th_full_pose = torch.cat([
            R.unsqueeze(0), self._model._MANO.th_hands_mean + th_full_hand_pose
            ], 1)
        th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
        th_full_pose = th_full_pose.view(1, -1, 3)
        root_rot = rodrigues_layer.batch_rodrigues(
            th_full_pose[:, 0]).view(1, 3, 3)

        th_v_shaped = torch.matmul(self._model._MANO.th_shapedirs,
                                   self._model._MANO.th_betas.transpose(1, 0)).permute(
                                       2, 0, 1) + self._model._MANO.th_v_template
        th_j = torch.matmul(self._model._MANO.th_J_regressor, th_v_shaped).repeat(
            1, 1, 1)

        root_j = th_j[:, 0, :].contiguous().view(1, 3, 1)
        th_results = []
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))
        angle_parents = [4294967295, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
        fingers = [1000000, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]

        relative_th_results = [[], [], [], [], []]
        # Rotate each part
        for i in range(15):
            i_val = int(i + 1)
            joint_rot = th_rot_map[:, (i_val - 1) * 9:i_val *
                                   9].contiguous().view(1, 3, 3)
            joint_j = th_j[:, i_val, :].contiguous().view(1, 3, 1)
            parent = make_list(angle_parents)[i_val]
            parent_j = th_j[:, parent, :].contiguous().view(1, 3, 1)
            joint_rel_transform = th_with_zeros(
                torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(
                torch.matmul(th_results[parent], joint_rel_transform))
            relative_th_results[fingers[i_val]].append(joint_rel_transform)

        for i in range(5):
            relative_th_results[i] = torch.cat(relative_th_results[i])
        relative_th_results = torch.stack(relative_th_results)

        Hs = torch.cat(th_results)
        Hs[:, :3, 3] = Hs[:, :3, 3] + T
        return Hs

    def get_fullhandpose(self, HR):
        th_full_hand_pose = HR.mm(self._model._MANO.th_selected_comps)
        return th_full_hand_pose
        

    def get_rot_map(self, HR, R):
        pose_representation = torch.cat((R, HR), 1)
        th_full_hand_pose = HR.mm(self._model._MANO.th_selected_comps)
        th_full_pose = torch.cat([
            R,
            self._model._MANO.th_hands_mean + th_full_hand_pose
        ], 1)
        th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
        th_full_pose = th_full_pose.view(1, -1, 3)
        root_rot = rodrigues_layer.batch_rodrigues(th_full_pose[:, 0]).view(1, 3, 3)

        return th_pose_map, th_rot_map, root_rot


    #def get_hand_all(self, root_rot, th_pose_map, th_rot_map, th_trans):
    def get_hand(self, th_full_hand_pose, R, th_trans):
        batch_size = len(th_trans)
        th_full_pose = torch.cat([
            R,
            self._model._MANO.th_hands_mean + th_full_hand_pose
        ], 1)
        th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
        th_full_pose = th_full_pose.view(batch_size, -1, 3)
        root_rot = rodrigues_layer.batch_rodrigues(th_full_pose[:, 0]).view(batch_size, 3, 3)

        # NOTE: WITH DEFAULT HAND SHAPE PARAMETERS:
        # th_v_shape is [batchsize, 778, 3] -> For baseline hand position
        th_v_shaped = torch.matmul(self._model._MANO.th_shapedirs,
                                   self._model._MANO.th_betas.transpose(1, 0)).permute(
                                       2, 0, 1) + self._model._MANO.th_v_template
        th_j = torch.matmul(self._model._MANO.th_J_regressor, th_v_shaped).repeat(
            batch_size, 1, 1)

        # NOTE: GET HAND MESH VERTICES: 778 vertices in 3D
        # th_v_posed -> [batchsize, 778, 3]
        # self.th_posedirs maps th_pose_map's 135 values to mesh vertices.
        th_v_posed = th_v_shaped + torch.matmul(
            self._model._MANO.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        # Final T pose with transformation done !

        # Global rigid transformation
        th_results = []

        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        th_results.append(th_with_zeros(torch.cat([root_rot, root_j], 2)))

        # Rotate each part
        for i in range(15):
            i_val = int(i + 1)
            joint_rot = th_rot_map[:, (i_val - 1) * 9:i_val *
                                   9].contiguous().view(batch_size, 3, 3)
            joint_j = th_j[:, i_val, :].contiguous().view(batch_size, 3, 1)
            parent = make_list(self._model._MANO.kintree_parents)[i_val]
            parent_j = th_j[:, parent, :].contiguous().view(batch_size, 3, 1)
            joint_rel_transform = th_with_zeros(
                torch.cat([joint_rot, joint_j - parent_j], 2))
            th_results.append(
                torch.matmul(th_results[parent], joint_rel_transform))
        th_results_global = th_results

        th_results2 = torch.zeros((batch_size, 4, 4, 16),
                                  dtype=root_j.dtype,
                                  device=root_j.device)
        for i in range(16):
            padd_zero = torch.zeros(1, dtype=th_j.dtype, device=th_j.device)
            joint_j = torch.cat(
                [th_j[:, i],
                 padd_zero.view(1, 1).repeat(batch_size, 1)], 1)
            tmp = torch.bmm(th_results[i], joint_j.unsqueeze(2))
            th_results2[:, :, :, i] = th_results[i] - th_pack(tmp)

        th_T = torch.matmul(th_results2, self._model._MANO.th_weights.transpose(0, 1))

        th_rest_shape_h = torch.cat([
            th_v_posed.transpose(2, 1),
            torch.ones((batch_size, 1, th_v_posed.shape[1]),
                       dtype=th_T.dtype,
                       device=th_T.device),
        ], 1)

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = torch.stack(th_results_global, dim=1)[:, :, :3, 3]
        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        if self._model._MANO.side == 'right':
            tips = torch.stack([
                th_verts[:, 745], th_verts[:, 317], th_verts[:, 444],
                th_verts[:, 556], th_verts[:, 673]
            ],
                               dim=1)
        else:
            tips = torch.stack([
                th_verts[:, 745], th_verts[:, 317], th_verts[:, 445],
                th_verts[:, 556], th_verts[:, 673]
            ],
                               dim=1)
        th_jtr = torch.cat([th_jtr, tips], 1)

        # Reorder joints to match visualization utilities
        th_jtr = torch.stack([
            th_jtr[:, 0], th_jtr[:, 13], th_jtr[:, 14], th_jtr[:, 15],
            th_jtr[:, 16], th_jtr[:, 1], th_jtr[:, 2], th_jtr[:, 3],
            th_jtr[:, 17], th_jtr[:, 4], th_jtr[:, 5], th_jtr[:, 6],
            th_jtr[:, 18], th_jtr[:, 10], th_jtr[:, 11], th_jtr[:, 12],
            th_jtr[:, 19], th_jtr[:, 7], th_jtr[:, 8], th_jtr[:, 9],
            th_jtr[:, 20]
        ],
                             dim=1)

        if th_trans is None or bool(torch.norm(th_trans) == 0):
            if self._model._MANO.center_idx is not None:
                center_joint = th_jtr[:, self._model._MANO.center_idx].unsqueeze(1)
                th_jtr = th_jtr - center_joint
                th_verts = th_verts - center_joint
        else:
            th_jtr = th_jtr + th_trans.unsqueeze(1)
            th_verts = th_verts + th_trans.unsqueeze(1)

        return th_verts, th_jtr

    def get_finger_distances(self, hand_verts, obj_verts):
        index_dists = _get_touching_distances(hand_verts[:, self._indexfinger_vertices], obj_verts).min().item()
        middle_dists = _get_touching_distances(hand_verts[:, self._middlefinger_vertices], obj_verts).min().item()
        fourth_dists = _get_touching_distances(hand_verts[:, self._fourthfinger_vertices], obj_verts).min().item()
        small_dists = _get_touching_distances(hand_verts[:, self._smallfinger_vertices], obj_verts).min().item()
        bigf_dists = _get_touching_distances(hand_verts[:, self._bigfinger_vertices], obj_verts).min().item()

        return bigf_dists, index_dists, middle_dists, fourth_dists, small_dists


    def interpolation_ahead(self, pose, optimize, eps):
        if optimize[0]:
            pose[0, 36:39] = pose[0, 36:39]*(1-eps[0]) + self.limit_bigfinger*eps[0]
        if optimize[1]:
            pose[0, 0:3] = pose[0, 0:3]*(1-eps[1]) + self.limit_index*eps[1]
        if optimize[2]:
            pose[0, 9:12] = pose[0, 9:12]*(1-eps[2]) + self.limit_middlefinger*eps[2]
        if optimize[3]:
            pose[0, 27:30] = pose[0, 27:30]*(1-eps[3]) + self.limit_fourth*eps[3]
        if optimize[4]:
            pose[0, 18:21] = pose[0, 18:21]*(1-eps[4]) + self.limit_small*eps[4]

        return pose


    def interpolation_behind(self, pose, optimize, eps):
        if optimize[0]:
            pose[0, 36:39] = pose[0, 36:39]*(1-eps)# + np.array([0, 0, 0])*eps
        if optimize[1]:
            pose[0, 0:3] = pose[0, 0:3]*(1-eps)# + np.array([0, 0, 0])*eps
        if optimize[2]:
            pose[0, 9:12] = pose[0, 9:12]*(1-eps)# + np.array([0, 0, 0])*eps
        if optimize[3]:
            pose[0, 27:30] = pose[0, 27:30]*(1-eps)# + np.array([0, 0, 0])*eps
        if optimize[4]:
            pose[0, 18:21] = pose[0, 18:21]*(1-eps)# + np.array([0, 0, 0])*eps

        return pose

    def get_intersections(self, hand_verts, obj_verts, obj_faces):
        finger_intersections = []

        obj_trimesh = trimesh.Trimesh(obj_verts, obj_faces)
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh=obj_trimesh)
        
        #vectors = points_3d[i].cpu().data.numpy()
        vectors = hand_verts[0][self._bigfinger_vertices].cpu().data.numpy()
        locations, index_ray, _ = intersector.intersects_location([[0, 0, 0]]*72, vectors)
        collisioned = np.unique(index_ray)
        outs = []
        for j in range(len(collisioned)):
            inds = np.where(index_ray == collisioned[j])[0]
            if not(np.all(locations[inds, -1] < vectors[collisioned[j], -1]) or np.all(locations[inds, -1] > vectors[collisioned[j], -1])):
                outs.append(collisioned[j])
        finger_intersections.append(len(outs))

        vectors = hand_verts[0][self._indexfinger_vertices].cpu().data.numpy()
        locations, index_ray, _ = intersector.intersects_location([[0, 0, 0]]*84, vectors)
        collisioned = np.unique(index_ray)
        outs = []
        for j in range(len(collisioned)):
            inds = np.where(index_ray == collisioned[j])[0]
            if not(np.all(locations[inds, -1] < vectors[collisioned[j], -1]) or np.all(locations[inds, -1] > vectors[collisioned[j], -1])):
                outs.append(collisioned[j])
        finger_intersections.append(len(outs))

        vectors = hand_verts[0][self._middlefinger_vertices].cpu().data.numpy()
        locations, index_ray, _ = intersector.intersects_location([[0, 0, 0]]*112, vectors)
        collisioned = np.unique(index_ray)
        outs = []
        for j in range(len(collisioned)):
            inds = np.where(index_ray == collisioned[j])[0]
            if not(np.all(locations[inds, -1] < vectors[collisioned[j], -1]) or np.all(locations[inds, -1] > vectors[collisioned[j], -1])):
                outs.append(collisioned[j])
        finger_intersections.append(len(outs))

        vectors = hand_verts[0][self._fourthfinger_vertices].cpu().data.numpy()
        locations, index_ray, _ = intersector.intersects_location([[0, 0, 0]]*112, vectors)
        collisioned = np.unique(index_ray)
        outs = []
        for j in range(len(collisioned)):
            inds = np.where(index_ray == collisioned[j])[0]
            if not(np.all(locations[inds, -1] < vectors[collisioned[j], -1]) or np.all(locations[inds, -1] > vectors[collisioned[j], -1])):
                outs.append(collisioned[j])
        finger_intersections.append(len(outs))

        vectors = hand_verts[0][self._smallfinger_vertices].cpu().data.numpy()
        locations, index_ray, _ = intersector.intersects_location([[0, 0, 0]]*115, vectors)
        collisioned = np.unique(index_ray)
        outs = []
        for j in range(len(collisioned)):
            inds = np.where(index_ray == collisioned[j])[0]
            if not(np.all(locations[inds, -1] < vectors[collisioned[j], -1]) or np.all(locations[inds, -1] > vectors[collisioned[j], -1])):
                outs.append(collisioned[j])
        finger_intersections.append(len(outs))

        return np.array(finger_intersections)

    def optimize_hand(self, handfullpose, R, T, obj_verts, obj_faces):
        #T = 1
        #q = trace_method(R)
        #q = pyquaternion.Quaternion(q)
        #axis = q.axis

        # TODO USE QUATERNION FOR INTERPOLATIONS?
        max_iter = 10
        eps = 0.2
        i = 0
        threshold_optimization = 0.001
        threshold_valid = 0.002
        while(i < max_iter):
            hand_verts, _ = self.get_hand(handfullpose, R, T)
            num_intersections = self.get_intersections(hand_verts, obj_verts, obj_faces)
            
            move_behind = num_intersections > 5
            if not np.max(move_behind):
                break

            handfullpose = self.interpolation_behind(handfullpose, move_behind, eps)
            i = i + 1

        resampled_obj_verts = pcu.sample_mesh_lloyd(obj_verts.astype(np.float32), obj_faces.astype(np.int32), 5000)
        #resampled_obj_verts = pcu.sample_mesh_lloyd(obj_verts.astype(np.float32), obj_faces.astype(np.int32), 50000)
        original_pose = handfullpose.clone()
        max_iter = 200
        i = 0
        threshold_optimization = 0.0015
        threshold_valid = 0.002
        previous_dists = np.array([np.Inf]*5)
        while(i < max_iter):
            hand_verts, _ = self.get_hand(handfullpose, R, T)
            current_dists = np.array(self.get_finger_distances(hand_verts, resampled_obj_verts))
            
            getcloser = previous_dists > current_dists
            getcloser &= current_dists > threshold_optimization

            eps = current_dists*8
            handfullpose = self.interpolation_ahead(handfullpose, getcloser, eps)
            previous_dists = current_dists.copy() - 0.0000001 # Threshold

            #print(current_dists)
            #print(getcloser)
            #print("")
            if not np.max(getcloser):
                break
            i = i + 1


        hand_verts, _ = self.get_hand(handfullpose, R, T)
        current_dists = np.array(self.get_finger_distances(hand_verts, resampled_obj_verts))
        
        isclose = current_dists < threshold_valid
        if not isclose[0]:
            handfullpose[0, 36:39] = original_pose[0, 36:39]
        if not isclose[1]:
            handfullpose[0, 0:3] = original_pose[0, 0:3]
        if not isclose[2]:
            handfullpose[0, 9:12] = original_pose[0, 9:12]
        if not isclose[3]:
            handfullpose[0, 27:30] = original_pose[0, 27:30]
        if not isclose[4]:
            handfullpose[0, 18:21] = original_pose[0, 18:21]

        return handfullpose, sum(isclose)

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
        #resampled_obj_verts = pcu.sample_mesh_lloyd(obj.astype(np.float32), obj_faces.astype(np.int32), 2000)
        #threshold = 0.001 # m
        threshold = 0.002 # m
        for i in range(len(finger_vertices)):
            dists = get_distance_vertices(resampled_obj_verts, hand[finger_vertices[i]])
            if np.min(dists) < threshold:
                faces = np.where(finger_vertices[i][np.argmin(dists)] == hand_faces)[0]
                normal = []
                for j in range(len(faces)):
                    normal.append(get_normal_face(hand[hand_faces[faces[j], 0]], hand[hand_faces[faces[j], 1]], hand[hand_faces[faces[j], 2]]))
                normal = np.mean(normal, 0) * 1e5 # Multiply by large number to avoid **2 going to zero
                normal = normal/np.sqrt((np.array(normal)**2).sum())
                torques.append([0, 0, 0])
                normals.append(normal)
                forces.append(normal)

        return forces, torques, normals


    def get_intersections_allhand(self, hand_verts, obj_verts, obj_faces):
        finger_intersections = []

        obj_trimesh = trimesh.Trimesh(obj_verts, obj_faces)
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh=obj_trimesh)
        
        #vectors = points_3d[i].cpu().data.numpy()
        vectors = hand_verts
        locations, index_ray, _ = intersector.intersects_location([[0, 0, 0]]*len(vectors), vectors)
        collisioned = np.unique(index_ray)
        outs = []
        for j in range(len(collisioned)):
            inds = np.where(index_ray == collisioned[j])[0]
            if not(np.all(locations[inds, -1] < vectors[collisioned[j], -1]) or np.all(locations[inds, -1] > vectors[collisioned[j], -1])):
                outs.append(collisioned[j])

        return len(outs)

if __name__ == '__main__':
    Test()
