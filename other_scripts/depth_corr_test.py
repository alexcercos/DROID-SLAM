import sys
sys.path.append("droid_slam")

import torch
import argparse

import droid_backends
import argparse
import open3d as o3d

from visualization import create_camera_actor
from lietorch import SE3
import numpy as np

import os
import math

from cuda_timer import CudaTimer

def view_reconstruction(cam_scale=0.05):
    # Load .npy files and convert to torch tensors

    # ---- Synthetic DISPS (Depth) ----
    H, W = 200, 200
    x = torch.linspace(-1, 1, W)
    y = torch.linspace(-1, 1, H)
    X, Y = torch.meshgrid(x, y, indexing='xy')

    # Wedge-shaped depth: minimum at center, increases toward sides
    disps = 2.0 + 0.5 * torch.abs(X)
    disps = disps.cuda()
    disps = disps.unsqueeze(0)

    pose = torch.tensor([0., 0., 0.,   0., 0., 0., 1.], device='cuda')
    poses = pose.unsqueeze(0)

    fx = fy = 214
    cx = W / 2.0
    cy = H / 2.0

    intrinsics = torch.tensor([fx, fy, cx, cy], dtype=torch.float32).cuda()
    intrinsics = intrinsics.unsqueeze(0)   # shape [1, 4]

    disps = disps.contiguous()

    # print(intrinsics.shape,disps.shape,poses.shape)

    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])

    B, H, W = disps.shape
    colors = torch.zeros((B, H, W, 3), device=disps.device, dtype=torch.float32)
    colors[..., 2] = 1.0      # Set blue channel

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
