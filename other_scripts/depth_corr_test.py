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
import matplotlib.pyplot as plt

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

def view_reconstruction(cam_scale=0.05, angle=0, print_matrix=False):

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

    R_np = coord_frame.get_rotation_matrix_from_xyz((0,math.radians(angle),0))
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

    coords0 = pops.coords_grid(H, W, device='cuda')
    coords, valid_mask = \
        pops.projective_transform(Gs, disps[None], intrinsics[None], ii, jj, return_depth=True)

    depth_jj = 1.0 / disps[jj] #[2, 16, 16]

    x = coords[..., 0] #[1, 2, 16, 16]
    y = coords[..., 1]

    B, F, H, W = x.shape

    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    wx = x - x0.float()
    wy = y - y0.float()

    #[1, 2, 16, 16]
    w00 = (1-wx)*(1-wy)
    w01 = (1-wx)*wy
    w10 = wx*(1-wy)
    w11 = wx*wy

    in_bounds = (
        (x0 >= 0) & (x1 <= W) &
        (y0 >= 0) & (y1 <= H)
    ).unsqueeze(-1)

    #[1, 2, 16, 16, 1]
    mask = valid_mask.bool() & in_bounds

    x0 = x0.clamp(0, W-1)
    y0 = y0.clamp(0, H-1)
    x1 = x1.clamp(0, W-1)
    y1 = y1.clamp(0, H-1)

    # bilinear sampling
    frame_idx = torch.arange(F, device="cuda")[None, :, None, None].expand(B, F, H, W)
    d00 = depth_jj[frame_idx, y0, x0]
    d01 = depth_jj[frame_idx, y1, x0]
    d10 = depth_jj[frame_idx, y0, x1]
    d11 = depth_jj[frame_idx, y1, x1]

    depth_sampled = w00*d00 + w01*d01 + w10*d10 + w11*d11

    # difference between projected depth and real depth at jj
    zdepth = 1.0 / coords[..., 2]
    depth_error = (depth_sampled - zdepth).abs().unsqueeze(-1)

    # depth_error = depth_error * mask

    K = 10.0
    corr = torch.exp(-K * depth_error) * mask

    if print_matrix:
        plt.figure(figsize=(12,12))
        ax = plt.gca()

        # Draw a grid
        for i in range(H+1):
            ax.axhline(i, color='black', linewidth=0.5)
        for j in range(W+1):
            ax.axvline(j, color='black', linewidth=0.5)

        # Put text in each cell
        for i in range(H):
            for j in range(W):
                er = depth_error[0, 0, i, j].item()
                s = depth_sampled[0,0,i,j].item()
                z = zdepth[0,0,i,j].item()
                c = corr[0,0,i,j].item()
                ax.text(j + 0.5, H - i - 0.5, f"{z:.2f} - {s:.2f}\n{er:.2f};{c:.2f}", 
                        ha='center', va='center', fontsize=4)

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        plt.show()

    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])

    mask = (disps > 0.25 * disps.mean())

    points_np = points[mask].cpu().numpy()

    B, H, W = disps.shape
    colors = torch.zeros((B, H, W, 3), device=disps.device, dtype=torch.float32)

    colormap = plt.colormaps.get_cmap('jet')
    colors_np = colors[mask].cpu().numpy()
    corr_masked = corr[mask[None].unsqueeze(-1)].cpu().numpy()
    jet_rgb = colormap(corr_masked)[:, :3]  # shape (N, 3)

    if print_matrix:
        print(corr_masked)

    colors_np[:, :] = jet_rgb
    colors[mask] = torch.from_numpy(colors_np).to(colors.device).float()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", type=float, default=0)
    # parser.add_argument("--filter_threshold", type=float, default=0.005)
    # parser.add_argument("--filter_count", type=int, default=3)
    parser.add_argument("--cam_scale", type=float, default=0.03)
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()

    view_reconstruction(cam_scale=args.cam_scale, angle=args.angle, print_matrix=args.print)
