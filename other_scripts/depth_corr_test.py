import sys
sys.path.append("droid_slam")

import torch
import argparse

import droid_backends
import argparse
import open3d as o3d

from visualization import create_camera_actor
import lietorch
from lietorch import SE3
import numpy as np

import os
import math

from cuda_timer import CudaTimer

import geom.projective_ops as pops

def rotation_matrix_to_quaternion(R):
    # R : numpy array (3x3)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]

    tr = m00 + m11 + m22

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2   # S = 4*qw
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S

    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S

    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S

    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    return np.array([qx, qy, qz, qw], dtype=np.float32)

def view_reconstruction(cam_scale=0.05):

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # ---- Synthetic DISPS (Depth) ----
    H, W = 16, 16
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # Wedge-shaped depth: minimum at center, increases toward sides
    disp1 = X/4 + 2
    disp2 = -X/4 + 2

    disp1 = disp1.cuda()
    disp2 = disp2.cuda()

    # Stack to get shape [2,16,16]
    disps = torch.stack([disp1, disp2], dim=0)

    pose = torch.tensor([0., 0., 0.5,   0., 0., 0., 1.], device='cuda')
    poses = pose.unsqueeze(0).repeat(2, 1)

    R_np = coord_frame.get_rotation_matrix_from_xyz((0,math.radians(-30),0))
    R = torch.tensor(R_np, dtype=torch.float32, device='cuda')

    # Rotate translation of pose #2
    pivot = torch.tensor([0., 0., 0.5], device='cuda')  # rotation center
    t = poses[1, :3] - pivot      # vector from pivot to camera
    t_rot = R @ t                 # rotate it
    poses[1, :3] = t_rot + pivot  # new camera position

    # Convert rotation matrix -> quaternion
    q_np = rotation_matrix_to_quaternion(R_np)
    poses[1, 3:] = torch.tensor(q_np, device='cuda')

    fx = 17.1
    fy = 17.1
    cx = W / 2.0
    cy = H / 2.0

    intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32, device='cuda')
    intrinsics = intrinsics.unsqueeze(0).repeat(2, 1)   # shape [2, 4]

    disps = disps.contiguous()

    Gs = lietorch.SE3(poses[None])
    ii = torch.tensor([0, 1], device="cuda", dtype=torch.long)
    jj = torch.tensor([1, 0], device="cuda", dtype=torch.long)

    coords, valid_mask = \
        pops.projective_transform(Gs, disps[None], intrinsics[None], ii, jj)

    print(coords.shape,valid_mask.shape)
    print(valid_mask[0, 0, :, :])

    mat = coords[0, 0, :, :]
    print(np.round(mat.cpu().numpy(), 2))

    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])

    B, H, W = disps.shape
    colors = torch.zeros((B, H, W, 3), device=disps.device, dtype=torch.float32)

    colors[0, ..., 2] = 1.0   # blue

    colors[1, ..., 0] = 1.0   # yellow
    colors[1, ..., 1] = 0.5

    mask = (disps > .25 * disps.mean())
    points_np = points[mask].cpu().numpy()
    colors_np = colors[mask].cpu().numpy()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_np)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(height=900, width=1600)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.add_geometry(point_cloud)

    vis.add_geometry(coord_frame)

    # get pose matrices as a nx4x4 numpy array
    pose_mats = SE3(poses).inv().matrix().cpu().numpy()

    ### add camera actor ###
    for i in range(len(poses)):
        cam_actor = create_camera_actor(True, cam_scale)
        cam_actor.transform(pose_mats[i])
        vis.add_geometry(cam_actor)

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("datapath", type=str, help="path to image directory")
    # parser.add_argument("--filter_threshold", type=float, default=0.005)
    # parser.add_argument("--filter_count", type=int, default=3)
    # parser.add_argument("--cam_scale", type=float, default=0.03)
    # args = parser.parse_args()

    view_reconstruction()
