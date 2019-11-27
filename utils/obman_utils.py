import json
import os
import traceback
import pickle

from joblib import Parallel, delayed
import numpy as np
import trimesh
from matplotlib import pyplot as plt

import socket
from subprocess import Popen
import shutil
import time
import tempfile

import pybullet as p
import skvideo
import skvideo.io as skvio
import cv2



# 3D model intersection:
def get_intersection(hand_verts, hand_faces, obj_verts, obj_faces):
    hand_mesh = trimesh.Trimesh(
        vertices=hand_verts, faces=hand_faces
    )
    obj_mesh = trimesh.Trimesh(
        vertices=obj_verts, faces=obj_faces
    )
    volume = intersect_vox(obj_mesh, hand_mesh, pitch=0.005)
    return volume


def intersect_vox(obj_mesh, hand_mesh, pitch=0.01):
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume



# 3D model simulations
def run_simulation(
    hand_verts,
    hand_faces,
    obj_verts,
    obj_faces,
    simulation_step=1 / 240,
    num_iterations=35,
    object_friction=3,
    hand_friction=3,
    hand_restitution=0,
    object_restitution=0.5,
    object_mass=1,
    verbose=False,
    vhacd_resolution=1000,
    vhacd_exe= '/home/ecorona/GHANds/v-hacd/bin/linux/testVHACD',
    wait_time=0,
    save_video=False,
    save_video_path=None,
    save_hand_path=None,
    save_obj_path=None,
    save_simul_folder=None,
    use_gui=False,
):
    if use_gui:
        conn_id = p.connect(p.GUI)
    else:
        conn_id = p.connect(p.DIRECT)

    hand_indicies = hand_faces.flatten().tolist()
    p.resetSimulation(physicsClientId=conn_id)
    p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=conn_id)
    p.setPhysicsEngineParameter(
        numSolverIterations=150, physicsClientId=conn_id
    )
    p.setPhysicsEngineParameter(
        fixedTimeStep=simulation_step, physicsClientId=conn_id
    )
    p.setGravity(0, 9.8, 0, physicsClientId=conn_id)

    # add hand
    base_tmp_dir = "tmp/objs"
    os.makedirs(base_tmp_dir, exist_ok=True)
    hand_tmp_fname = tempfile.mktemp(suffix=".obj", dir=base_tmp_dir)
    save_obj(hand_tmp_fname, hand_verts, hand_faces)

    if save_hand_path is not None:
        shutil.copy(hand_tmp_fname, save_hand_path)

    hand_collision_id = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=hand_tmp_fname,
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
        indices=hand_indicies,
        physicsClientId=conn_id,
    )
    hand_visual_id = p.createVisualShape(
        p.GEOM_MESH,
        fileName=hand_tmp_fname,
        rgbaColor=[0, 0, 1, 1],
        specularColor=[0, 0, 1],
        physicsClientId=conn_id,
    )

    hand_body_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=hand_collision_id,
        baseVisualShapeIndex=hand_visual_id,
        physicsClientId=conn_id,
    )

    p.changeDynamics(
        hand_body_id,
        -1,
        lateralFriction=hand_friction,
        restitution=hand_restitution,
        physicsClientId=conn_id,
    )

    obj_tmp_fname = tempfile.mktemp(suffix=".obj", dir=base_tmp_dir)
    os.makedirs(base_tmp_dir, exist_ok=True)
    # Save object obj
    if save_obj_path is not None:
        final_obj_tmp_fname = tempfile.mktemp(suffix=".obj", dir=base_tmp_dir)
        save_obj(final_obj_tmp_fname, obj_verts, obj_faces)
        shutil.copy(final_obj_tmp_fname, save_obj_path)
    # Get obj center of mass
    obj_center_mass = np.mean(obj_verts, axis=0)
    obj_verts -= obj_center_mass
    # add object
    use_vhacd = True

    if use_vhacd:
        if verbose:
            print("Computing vhacd decomposition")
            time1 = time.time()
        # convex hull decomposition
        save_obj(obj_tmp_fname, obj_verts, obj_faces)

        if not vhacd(obj_tmp_fname, vhacd_exe, resolution=vhacd_resolution):
            raise RuntimeError(
                "Cannot compute convex hull "
                "decomposition for {}".format(obj_tmp_fname)
            )
        else:
            print("Succeeded vhacd decomp")

        obj_collision_id = p.createCollisionShape(
            p.GEOM_MESH, fileName=obj_tmp_fname, physicsClientId=conn_id
        )
        if verbose:
            time2 = time.time()
            print(
                "Computed v-hacd decomposition at res {} {:.6f} s".format(
                    vhacd_resolution, (time2 - time1)
                )
            )
    else:
        obj_collision_id = p.createCollisionShape(
            p.GEOM_MESH, vertices=obj_verts, physicsClientId=conn_id
        )

    obj_visual_id = p.createVisualShape(
        p.GEOM_MESH,
        fileName=obj_tmp_fname,
        rgbaColor=[1, 0, 0, 1],
        specularColor=[1, 0, 0],
        physicsClientId=conn_id,
    )
    obj_body_id = p.createMultiBody(
        baseMass=object_mass,
        basePosition=obj_center_mass,
        baseCollisionShapeIndex=obj_collision_id,
        baseVisualShapeIndex=obj_visual_id,
        physicsClientId=conn_id,
    )

    p.changeDynamics(
        obj_body_id,
        -1,
        lateralFriction=object_friction,
        restitution=object_restitution,
        physicsClientId=conn_id,
    )

    # simulate for several steps
    if save_video:
        images = []
        if use_gui:
            renderer = p.ER_BULLET_HARDWARE_OPENGL
        else:
            renderer = p.ER_TINY_RENDERER

    for step_idx in range(num_iterations):
        p.stepSimulation(physicsClientId=conn_id)
        if save_video:
            img = take_picture(renderer, conn_id=conn_id)
            images.append(img)
        if save_simul_folder:
            hand_step_path = os.path.join(
                save_simul_folder, "{:08d}_hand.obj".format(step_idx)
            )
            shutil.copy(hand_tmp_fname, hand_step_path)
            obj_step_path = os.path.join(
                save_simul_folder, "{:08d}_obj.obj".format(step_idx)
            )
            pos, orn = p.getBasePositionAndOrientation(
                obj_body_id, physicsClientId=conn_id
            )
            mat = np.reshape(p.getMatrixFromQuaternion(orn), (3, 3))
            obj_verts_t = pos + np.dot(mat, obj_verts.T).T
            save_obj(obj_step_path, obj_verts_t, obj_faces)
        time.sleep(wait_time)

    if save_video:
        write_video(images, save_video_path)
        print("Saved gif to {}".format(save_video_path))
    pos_end = p.getBasePositionAndOrientation(
        obj_body_id, physicsClientId=conn_id
    )[0]

    #if use_vhacd:
    #    os.remove(obj_tmp_fname)
    if save_obj_path is not None:
        os.remove(final_obj_tmp_fname)
    os.remove(hand_tmp_fname)
    distance = np.linalg.norm(pos_end - obj_center_mass)
    p.disconnect(physicsClientId=conn_id)
    return distance

def save_obj(filename, verticies, faces):
    with open(filename, "w") as fp:
        for v in verticies:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write("f %d %d %d\n" % (f[0], f[1], f[2]))


def vhacd(
    filename,
    vhacd_path,
    resolution=1000,
    concavity=0.001,
    planeDownsampling=4,
    convexhullDownsampling=4,
    alpha=0.05,
    beta=0.0,
    maxhulls=1024,
    pca=0,
    mode=0,
    maxNumVerticesPerCH=64,
    minVolumePerCH=0.0001,
):

    cmd_line = (
        '"{}" --input "{}" --resolution {} --concavity {:g} '
        "--planeDownsampling {} --convexhullDownsampling {} "
        "--alpha {:g} --beta {:g} --maxhulls {:g} --pca {:b} "
        "--mode {:b} --maxNumVerticesPerCH {} --minVolumePerCH {:g} "
        '--output "{}" --log "/dev/null"'.format(
            vhacd_path,
            filename,
            resolution,
            concavity,
            planeDownsampling,
            convexhullDownsampling,
            alpha,
            beta,
            maxhulls,
            pca,
            mode,
            maxNumVerticesPerCH,
            minVolumePerCH,
            filename,
        )
    )
    print(cmd_line)

    devnull = open(os.devnull, "wb")
    vhacd_process = Popen(
        cmd_line,
        bufsize=-1,
        close_fds=True,
        shell=True,
        stdout=devnull,
        stderr=devnull,
    )
    return 0 == vhacd_process.wait()


