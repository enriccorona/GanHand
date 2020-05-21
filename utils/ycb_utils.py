import numpy as np
import torch

def vertices_reprojection(vertices, rt, k):
    p = np.matmul(k, np.matmul(rt[:3,0:3], vertices.T) + rt[:3,3].reshape(-1,1))
    p[0] = p[0] / (p[2] + 1e-5)
    p[1] = p[1] / (p[2] + 1e-5)
    return p[:2].T


def vertices_final_projection(p, intrinsics):
    p = np.matmul(intrinsics, p.T)
    p[0] = p[0] / (p[2] + 1e-5)
    p[1] = p[1] / (p[2] + 1e-5)
    return p[:2].T


def get_plane_collision(verts_hand, plane):
    point_1 = torch.FloatTensor(([0, 0, plane[3]/plane[2], 1])).cuda()
    dot = torch.matmul(torch.cat((verts_hand, torch.ones((len(verts_hand),1)).cuda()), 1) - point_1, torch.FloatTensor(plane).cuda())
    return (dot > 0).float()

def get_plane_distance(verts_hand, plane):
    point_1 = torch.FloatTensor(([0, 0, plane[3]/plane[2], 1])).cuda()
    dot = torch.matmul(torch.cat((verts_hand, torch.ones((len(verts_hand),1)).cuda()), 1) - point_1, plane)
    return (dot > 0).float(), dot
