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

from manopth.manolayer import ManoLayer
import transforms3d
from .MANO_indices import *

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

def rmat_to_axisangle(R): 
    num = torch.trace(R) - 1 
    num = num/2

    # Avoid nans by limit between -1 and 1
    num = torch.min(num, torch.FloatTensor([1.0]).cuda())
    num = torch.max(num, torch.FloatTensor([-1.0]).cuda())

    theta = torch.acos(num) 
    vec = [R[2, 1] - R[1, 2], 
           R[0, 2] - R[2, 0], 
           R[1, 0] - R[0, 1]] 
    vec = torch.stack(vec) 
    w = vec/(2*torch.sin(theta)) 
    return w*theta 

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

    G = np.zeros([6, num_cols])
    for i in range(num_forces):
        G[:3,i] = forces[:,i]
        G[3:,i] = forces[:,i] # ZEROS
        #G[3:,i] = params.torque_scaling * torques[:,i]

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

def batch_get_normal_face(ps1, ps2, ps3):
    U = ps2 - ps1
    V = ps3 - ps1
    Nx = U[:, 1]*V[:, 2] - U[:, 2]*V[:, 1]
    Ny = U[:, 2]*V[:, 0] - U[:, 0]*V[:, 2]
    Nz = U[:, 0]*V[:, 1] - U[:, 1]*V[:, 0]
    return torch.cat((Nx.unsqueeze(1), Ny.unsqueeze(1), Nz.unsqueeze(1)), 1)

def get_distance_vertices(obj, hand):
    distances = []
    n1 = len(hand)
    n2 = len(obj)

    matrix1 = hand[np.newaxis].repeat(n2, 0)
    matrix2 = obj[:, np.newaxis].repeat(n1, 1)
    dists = np.sqrt(((matrix1-matrix2)**2).sum(-1))
    return dists.min(0)


# Important variables regarding MANO structure, needed for optimization:
# FINGER LIMIT ANGLE:

def get_joint_Hs(HR, R, T):
    th_full_hand_pose = HR.unsqueeze(0).mm(MANO.th_selected_comps)
    th_full_pose = torch.cat([
        R.unsqueeze(0), MANO.th_hands_mean + th_full_hand_pose
        ], 1)
    th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
    th_full_pose = th_full_pose.view(1, -1, 3)
    root_rot = rodrigues_layer.batch_rodrigues(
        th_full_pose[:, 0]).view(1, 3, 3)

    th_v_shaped = torch.matmul(MANO.th_shapedirs,
                               MANO.th_betas.transpose(1, 0)).permute(
                                   2, 0, 1) + MANO.th_v_template
    th_j = torch.matmul(MANO.th_J_regressor, th_v_shaped).repeat(
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


def get_Hs_fingers(HR, R, T):
    th_full_hand_pose = HR.unsqueeze(0).mm(MANO.th_selected_comps)
    th_full_pose = torch.cat([
        R.unsqueeze(0), MANO.th_hands_mean + th_full_hand_pose
        ], 1)
    th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
    th_full_pose = th_full_pose.view(1, -1, 3)
    root_rot = rodrigues_layer.batch_rodrigues(
        th_full_pose[:, 0]).view(1, 3, 3)

    th_v_shaped = torch.matmul(MANO.th_shapedirs,
                               MANO.th_betas.transpose(1, 0)).permute(
                                   2, 0, 1) + MANO.th_v_template
    th_j = torch.matmul(MANO.th_J_regressor, th_v_shaped).repeat(
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


def get_fullhandpose(HR):
    th_full_hand_pose = HR.mm(MANO.th_selected_comps)
    return th_full_hand_pose
    

def get_rot_map(HR, R):
    pose_representation = torch.cat((R, HR), 1)
    th_full_hand_pose = HR.mm(MANO.th_selected_comps)
    th_full_pose = torch.cat([
        R,
        MANO.th_hands_mean + th_full_hand_pose
    ], 1)
    th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
    th_full_pose = th_full_pose.view(1, -1, 3)
    root_rot = rodrigues_layer.batch_rodrigues(th_full_pose[:, 0]).view(1, 3, 3)

    return th_pose_map, th_rot_map, root_rot


#def get_hand_all(root_rot, th_pose_map, th_rot_map, th_trans):
def get_hand(th_full_hand_pose, R, th_trans):
    batch_size = len(th_trans)
    th_full_pose = torch.cat([
        R,
        MANO.th_hands_mean + th_full_hand_pose
    ], 1)
    th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
    th_full_pose = th_full_pose.view(batch_size, -1, 3)
    root_rot = rodrigues_layer.batch_rodrigues(th_full_pose[:, 0]).view(batch_size, 3, 3)

    # NOTE: WITH DEFAULT HAND SHAPE PARAMETERS:
    # th_v_shape is [batchsize, 778, 3] -> For baseline hand position
    th_v_shaped = torch.matmul(MANO.th_shapedirs,
                               MANO.th_betas.transpose(1, 0)).permute(
                                   2, 0, 1) + MANO.th_v_template
    th_j = torch.matmul(MANO.th_J_regressor, th_v_shaped).repeat(
        batch_size, 1, 1)

    # NOTE: GET HAND MESH VERTICES: 778 vertices in 3D
    # th_v_posed -> [batchsize, 778, 3]
    th_v_posed = th_v_shaped + torch.matmul(
        MANO.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
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
        parent = make_list(MANO.kintree_parents)[i_val]
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

    th_T = torch.matmul(th_results2, MANO.th_weights.transpose(0, 1))

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
    if MANO.side == 'right':
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
        if MANO.center_idx is not None:
            center_joint = th_jtr[:, MANO.center_idx].unsqueeze(1)
            th_jtr = th_jtr - center_joint
            th_verts = th_verts - center_joint
    else:
        th_jtr = th_jtr + th_trans.unsqueeze(1)
        th_verts = th_verts + th_trans.unsqueeze(1)

    return th_verts, th_jtr

# HAND optimization functions
# NOTE: UPDATE FUNCTIONS!! THEY WILL KEEP CHANGING!!

def optimize_hand(handfullpose, R, T, obj_verts, step=20):
    # Links between joints are:
    # Thumb: 0, 1, 2, 3, 4
    # Index: 0, 5, 6, 7, 8
    # Middle: 0, 9, 10, 11, 12
    # Fourth: 0, 13, 14, 15, 16
    # Small: 0, 17, 18, 19, 20

    # Thumb has two degrees of freedom when closing. 
    # Horizontal DOF is given by prediction and vertical DOF (Grasp direction) is given by following R:
    #Rthumb_vert = torch.FloatTensor([[ 0.8196, -0.4868, -0.3022],
    #                                [-0.1282, 0.3583, -0.9247],
    #                                [0.5585, 0.7966, 0.2313]]).cuda()
    #thumb_pred = rodrigues_layer.batch_rodrigues(handfullpose[:, 36:39]).view(3, 3)
    ## Limit the actual rotation because this was from open to close. Let's interpolate

    # SLERP between the quaternion of that RM and quat(1, 0, 0, 0) are:
    # 0.3:
    #array([[ 0.90412033, -0.31296735, -0.29089149],
   #[-0.01381291,  0.65903709, -0.75198359],
   #[ 0.42705459,  0.68390171,  0.59152585]])

    # 0.5:
    #array([[ 0.94920308, -0.20209229, -0.24118918],
   #[ 0.02896454,  0.8193583 , -0.57254959],
   #[ 0.31332821,  0.5364799 ,  0.78359093]])

    # 0.7:
    #array([[ 0.98125081, -0.10487662, -0.16170265],
   #[ 0.040975  ,  0.93332498, -0.35668689],
   #[ 0.18832923,  0.34337353,  0.92012321]])

    #Rmat_open = torch.matmul(torch.inverse(Rthumb_vert), thumb_pred)
    #Rmat_closed = torch.matmul(Rthumb_vert, thumb_pred)

    thumb_pred = rodrigues_layer.batch_rodrigues(handfullpose[:, 36:39]).view(3, 3)

    # NOTE: THIS IS NOT ACCURATE WHEN thumb_pred IS SUPER CLOSE FROM THE RESTING THUMB POSITION
    # NEITHER WHEN ROTATION (composition) ROTATION IS SUPER CLOSE .. "" ""
    # IN THAT CASE IT JUST DOESNT OPTIMIZES THUMB
    # TO SOLVE THIS - I SHOULD CHECK ANGLE FROM handfullpose[:, 36:39] AND SEE IF IT'S FURTHER THAN limit_bigfinger or too close

    #Let's get to double the rotation. We'll limit this to a maximum difference with respect root thumb position

    #Rmat_closed = torch.matmul(thumb_pred, thumb_pred)
    #Rmat_closed = torch.min(Rmat_closed, torch.FloatTensor([0.99999]).cuda())
    #Rmat_closed = torch.max(Rmat_closed, torch.FloatTensor([-0.99999]).cuda())

    # NOTE: Way 2 of doing it. Consider a maximum angle of rotation for thumb. We'll just 
    # Limit rotation matrix to it - converting to euler, scaling and back
    thumb_pred = torch.min(thumb_pred, torch.FloatTensor([0.99999]).cuda())
    thumb_pred = torch.max(thumb_pred, torch.FloatTensor([-0.99999]).cuda())
    eu = transforms3d.euler.mat2euler(thumb_pred.cpu().data.numpy())
    eu = np.array(eu)
    #eu = eu*2.0/np.sqrt((eu**2).sum())
    eu = eu*1.3043172907561458/np.sqrt((eu**2).sum())

    Rmat_closed = transforms3d.euler.euler2mat(eu[0], eu[1], eu[2])
    Rmat_closed = torch.FloatTensor(Rmat_closed).cuda()
    Rmat_closed = torch.min(Rmat_closed, torch.FloatTensor([0.99999]).cuda())
    Rmat_closed = torch.max(Rmat_closed, torch.FloatTensor([-0.99999]).cuda())

    # Limit was found by:
    # rotm = rodrigues_layer.batch_rodrigues(limit_bigfinger.unsqueeze(0)).view(3, 3)
    # angle = transforms3d.euler.mat2euler(rotm.cpu().data.numpy())
    # 1.3043172907561458 = np.sqrt((np.array(angle)**2).sum())

    #thumb_axisangle_open = rmat_to_axisangle(Rmat_open)
    thumb_axisangle_closed = rmat_to_axisangle(Rmat_closed)

    #handfullpose_open = handfullpose.clone()
    #handfullpose_open[0, 36:39] = torch.FloatTensor([0, 0, 0])
    ##handfullpose_open[0, 36:39] = thumb_axisangle_open #torch.FloatTensor([0, 0, 0])
    #handfullpose_open[0, 0:3] = torch.FloatTensor([0, 0, 0])
    #handfullpose_open[0, 9:12] = torch.FloatTensor([0, 0, 0])
    #handfullpose_open[0, 27:30] = torch.FloatTensor([0, 0, 0])
    #handfullpose_open[0, 18:21] = torch.FloatTensor([0, 0, 0])

    #handfullpose_closed = handfullpose.clone()
    #handfullpose_closed[0, 36:39] = thumb_axisangle_closed #limit_bigfinger_right
    #handfullpose_closed[0, 0:3] = limit_index_right
    #handfullpose_closed[0, 9:12] = limit_middlefinger_right
    #handfullpose_closed[0, 27:30] = limit_fourth_right
    #handfullpose_closed[0, 18:21] = limit_small_right

    #hand_verts_open, hand_joints_open = get_hand(handfullpose_open, R, T)
    #hand_verts_closed, hand_joints_closed = get_hand(handfullpose_closed, R, T)

    #knuckle_joints_open = hand_joints_open[0, [1, 5, 9, 13, 17]]
    #knuckle_joints_closed = hand_joints_closed[0, [1, 5, 9, 13, 17]]

    # OPTIMIZE EACH FINGER INDEPENDENTLY:
    handfullpose_converged = handfullpose.clone()
    touching_indexs = 0

    loss_distance = torch.FloatTensor([0]).cuda()
    loss_angle = torch.FloatTensor([0]).cuda()

    num_samples = 1000//step + 1
    inds = torch.linspace(0, 1, num_samples).cuda().unsqueeze(1)
    handfullpose_repeated = handfullpose_converged.clone().repeat(num_samples, 1)
    handfullpose_repeated[:, 36:39] = thumb_axisangle_closed.unsqueeze(0)*inds
    handfullpose_repeated[:, 0:3] = limit_index_right.unsqueeze(0)*inds
    handfullpose_repeated[:, 9:12] = limit_middlefinger_right.unsqueeze(0)*inds
    handfullpose_repeated[:, 27:30] = limit_fourth_right.unsqueeze(0)*inds
    handfullpose_repeated[:, 18:21] = limit_small_right.unsqueeze(0)*inds

    meshes, _ = get_hand(handfullpose_repeated, R.repeat(num_samples, 1), T.repeat(num_samples, 1))

    relevant_verts_thumb = meshes[:, bigfinger_vertices]
    relevant_verts_index = meshes[:, indexfinger_vertices]
    relevant_verts_middle = meshes[:, middlefinger_vertices]
    relevant_verts_fourth = meshes[:, fourthfinger_vertices]
    relevant_verts_small = meshes[:, smallfinger_vertices]

   # from IPython import embed
   # embed()

    # Thumb: 
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_thumb, obj_verts)
    loss_distance = loss_distance + distance_to_minimize.mean()
    if converged:
        handfullpose_converged[0, 36:39] = thumb_axisangle_closed*inds[vertex_solution]
        loss_angle = loss_angle + (handfullpose[0, 36:39] - thumb_axisangle_closed*inds[vertex_solution])**2
        touching_indexs += 1
    else:
        handfullpose_converged[0, 36:39] = thumb_axisangle_closed

    # Index:
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_index, obj_verts)
    loss_distance = loss_distance + distance_to_minimize.mean()
    if converged:
        handfullpose_converged[0, 0:3] = limit_index_right*inds[vertex_solution]
        loss_angle = loss_angle + (handfullpose[0, 0:3] - limit_index_right*inds[vertex_solution])**2
        touching_indexs += 1
    else:
        handfullpose_converged[0, 0:3] = limit_index_right

    # Middle:
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_middle, obj_verts)
    loss_distance = loss_distance + distance_to_minimize.mean()
    if converged:
        handfullpose_converged[0, 9:12] = limit_middlefinger_right*inds[vertex_solution]
        loss_angle = loss_angle + (handfullpose[0, 9:12] - limit_middlefinger_right*inds[vertex_solution])**2
        touching_indexs += 1
    else:
        handfullpose_converged[0, 9:12] = limit_middlefinger_right

    # Fourth:
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_fourth, obj_verts)
    loss_distance = loss_distance + distance_to_minimize.mean()
    if converged:
        handfullpose_converged[0, 27:30] = limit_fourth_right*inds[vertex_solution]
        loss_angle = loss_angle + (handfullpose[0, 27:30] - limit_fourth_right*inds[vertex_solution])**2
        touching_indexs += 1
    else:
        handfullpose_converged[0, 27:30] = limit_fourth_right

    # Small:
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_small, obj_verts)
    loss_distance = loss_distance + distance_to_minimize.mean()
    if converged:
        handfullpose_converged[0, 18:21] = limit_small_right*inds[vertex_solution]
        loss_angle = loss_angle + (handfullpose[0, 18:21] - limit_small_right*inds[vertex_solution])**2
        touching_indexs += 1
    else:
        handfullpose_converged[0, 18:21] = limit_small_right


    #####                           #####
    ##### SECOND JOINT OPTIMIZATION #####
    #####                           #####


    loss_angle_secondjoint = torch.FloatTensor([0]).cuda()
    touching_indexs = 0

    num_samples = 1000//step + 1
    inds = torch.linspace(0, 1, num_samples).cuda().unsqueeze(1)
    handfullpose_repeated = handfullpose_converged.clone().repeat(num_samples, 1)
    handfullpose_repeated[:, 39:42] = limit_secondjoint_bigfinger_right.unsqueeze(0)*inds
    handfullpose_repeated[:, 3:6] = limit_secondjoint_index_right.unsqueeze(0)*inds
    handfullpose_repeated[:, 12:15] = limit_secondjoint_middlefinger_right.unsqueeze(0)*inds
    handfullpose_repeated[:, 30:33] = limit_secondjoint_fourth_right.unsqueeze(0)*inds
    handfullpose_repeated[:, 21:24] = limit_secondjoint_small_right.unsqueeze(0)*inds

    meshes, _ = get_hand(handfullpose_repeated, R.repeat(num_samples, 1), T.repeat(num_samples, 1))

    relevant_verts_secondjoint_thumb = meshes[:, bigfinger_secondjoint_vertices]
    relevant_verts_secondjoint_index = meshes[:, indexfinger_secondjoint_vertices]
    relevant_verts_secondjoint_middle = meshes[:, middlefinger_secondjoint_vertices]
    relevant_verts_secondjoint_fourth = meshes[:, fourthfinger_secondjoint_vertices]
    relevant_verts_secondjoint_small = meshes[:, smallfinger_secondjoint_vertices]


    # Thumb: 
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_secondjoint_thumb, obj_verts)
    if converged:
        handfullpose_converged[0, 39:42] = limit_secondjoint_bigfinger_right*inds[vertex_solution]
        loss_angle_secondjoint = loss_angle_secondjoint + (handfullpose[0, 39:42] - limit_secondjoint_bigfinger_right*inds[vertex_solution])**2
        touching_indexs += 1

    # Index:
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_secondjoint_index, obj_verts)
    if converged:
        handfullpose_converged[0, 3:6] = limit_secondjoint_index_right*inds[vertex_solution]
        loss_angle_secondjoint = loss_angle_secondjoint + (handfullpose[0, 3:6] - limit_secondjoint_index_right*inds[vertex_solution])**2
        touching_indexs += 1

    # Middle:
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_secondjoint_middle, obj_verts)
    if converged:
        handfullpose_converged[0, 12:15] = limit_secondjoint_middlefinger_right*inds[vertex_solution]
        loss_angle_secondjoint = loss_angle_secondjoint + (handfullpose[0, 12:15] - limit_secondjoint_middlefinger_right*inds[vertex_solution])**2
        touching_indexs += 1

    # Fourth:
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_secondjoint_fourth, obj_verts)
    if converged:
        handfullpose_converged[0, 30:33] = limit_secondjoint_fourth_right*inds[vertex_solution]
        loss_angle_secondjoint = loss_angle_secondjoint + (handfullpose[0, 30:33] - limit_secondjoint_fourth_right*inds[vertex_solution])**2
        touching_indexs += 1

    # Small:
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_secondjoint_small, obj_verts)
    if converged:
        handfullpose_converged[0, 21:24] = limit_secondjoint_small_right*inds[vertex_solution]
        loss_angle_secondjoint = loss_angle_secondjoint + (handfullpose[0, 21:24] - limit_secondjoint_small_right*inds[vertex_solution])**2
        touching_indexs += 1



    #####                          #####
    ##### THIRD JOINT OPTIMIZATION #####
    #####                          #####

    loss_angle_thirdjoint = torch.FloatTensor([0]).cuda()

    touching_indexs = 0

    num_samples = 1000//step + 1
    inds = torch.linspace(0, 1, num_samples).cuda().unsqueeze(1)
    handfullpose_repeated = handfullpose_converged.clone().repeat(num_samples, 1)
    handfullpose_repeated[:, 42:45] = limit_thirdjoint_bigfinger_right.unsqueeze(0)*inds
    handfullpose_repeated[:, 6:9] = limit_thirdjoint_index_right.unsqueeze(0)*inds
    handfullpose_repeated[:, 15:18] = limit_thirdjoint_middlefinger_right.unsqueeze(0)*inds
    handfullpose_repeated[:, 33:36] = limit_thirdjoint_fourth_right.unsqueeze(0)*inds
    handfullpose_repeated[:, 24:27] = limit_thirdjoint_small_right.unsqueeze(0)*inds

    meshes, _ = get_hand(handfullpose_repeated, R.repeat(num_samples, 1), T.repeat(num_samples, 1))

    relevant_verts_thirdjoint_thumb = meshes[:, bigfinger_thirdjoint_vertices]
    relevant_verts_thirdjoint_index = meshes[:, indexfinger_thirdjoint_vertices]
    relevant_verts_thirdjoint_middle = meshes[:, middlefinger_thirdjoint_vertices]
    relevant_verts_thirdjoint_fourth = meshes[:, fourthfinger_thirdjoint_vertices]
    relevant_verts_thirdjoint_small = meshes[:, smallfinger_thirdjoint_vertices]

    # Thumb: 
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_thirdjoint_thumb, obj_verts)
    if converged:
        handfullpose_converged[0, 42:45] = limit_thirdjoint_bigfinger_right*inds[vertex_solution]
        loss_angle_thirdjoint = loss_angle_thirdjoint + (handfullpose[0, 42:45] - limit_thirdjoint_bigfinger_right*inds[vertex_solution])**2
        touching_indexs += 1

    # Index:
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_thirdjoint_index, obj_verts)
    if converged:
        handfullpose_converged[0, 6:9] = limit_thirdjoint_index_right*inds[vertex_solution]
        loss_angle_thirdjoint = loss_angle_thirdjoint + (handfullpose[0, 6:9] - limit_thirdjoint_index_right*inds[vertex_solution])**2
        touching_indexs += 1

    # Middle:
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_thirdjoint_middle, obj_verts)
    if converged:
        handfullpose_converged[0, 15:18] = limit_thirdjoint_middlefinger_right*inds[vertex_solution]
        loss_angle_thirdjoint = loss_angle_thirdjoint + (handfullpose[0, 15:18] - limit_thirdjoint_middlefinger_right*inds[vertex_solution])**2
        touching_indexs += 1

    # Fourth:
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_thirdjoint_fourth, obj_verts)
    if converged:
        handfullpose_converged[0, 33:36] = limit_thirdjoint_fourth_right*inds[vertex_solution]
        loss_angle_thirdjoint = loss_angle_thirdjoint + (handfullpose[0, 33:36] - limit_thirdjoint_fourth_right*inds[vertex_solution])**2
        touching_indexs += 1

    # Small:
    distance_to_minimize, vertex_solution, converged = get_optimization_angle(relevant_verts_thirdjoint_small, obj_verts)
    if converged:
        handfullpose_converged[0, 24:27] = limit_thirdjoint_small_right*inds[vertex_solution]
        loss_angle_thirdjoint = loss_angle_thirdjoint + (handfullpose[0, 24:27] - limit_thirdjoint_small_right*inds[vertex_solution])**2
        touching_indexs += 1

    return handfullpose_converged, touching_indexs, loss_distance, loss_angle.mean(), loss_angle_secondjoint.mean(), loss_angle_thirdjoint.mean()


def get_optimization_angle(arc_points, obj_verts):
    if type(obj_verts) is not torch.Tensor:
        obj_verts = torch.FloatTensor(obj_verts).cuda()
    obj_verts = obj_verts.unsqueeze(1).unsqueeze(1)
    arc_points = arc_points.unsqueeze(0)

    dists = obj_verts - arc_points
    eu_dists = torch.sqrt((dists**2).sum(-1))
    min_dist = eu_dists.min()
    #min_dist = eu_dists.min(0)[0].min(0)[0]

    threshold = 0.002
    #threshold = 0.001
    #threshold = 0.0001
    eu_dists = eu_dists.min(0)[0]
    solutions = eu_dists < threshold
    #solutions[0, :] = 0 # We want hand closer

    earliest_in_arc = solutions.cpu().data.numpy().argmax(0)
    earliest_in_arc[earliest_in_arc == 0] = 999
    vertex_solution = earliest_in_arc.min()

    if vertex_solution == 999:
        vertex_solution = 0

    converged = solutions.max().cpu().data.numpy()==1
    return min_dist, vertex_solution, converged
