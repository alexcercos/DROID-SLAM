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

import matplotlib as plt
import matplotlib.colors as mcolors

from cuda_timer import CudaTimer
import geom.projective_ops as pops

def view_reconstruction(datapath: str, filter_thresh=0.005, filter_count=2, cam_scale=0.05):
    # Load .npy files and convert to torch tensors
    # images = torch.from_numpy(np.load(f"{datapath}/images.npy")).cuda()[..., ::2, ::2]
    disps = torch.from_numpy(np.load(f"{datapath}/disps.npy")).cuda()[..., ::2, ::2]
    poses = torch.from_numpy(np.load(f"{datapath}/poses.npy")).cuda()
    intrinsics = 4 * torch.from_numpy(np.load(f"{datapath}/intrinsics.npy")).cuda()

    disps = disps[3:5]
    poses = poses[3:5]
    intrinsics = intrinsics[:2]

    disps = disps.contiguous()


    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
    
    # index = torch.arange(len(disps), device="cuda")
    # thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

    # with CudaTimer("filter"):
    #     counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    # mask = (counts >= filter_count) & (disps > .25 * disps.mean())
    mask = (disps > .25 * disps.mean())
    points_np = points[mask].cpu().numpy()

    # --- Inverse depth coloring --- *****************************

    print(disps.shape,poses.shape,intrinsics.shape)

    Gs = SE3(poses[None])
    #TODO en real habria que utilizar los indices que vengan dados
    ii = torch.tensor([0, 1], device="cuda", dtype=torch.long)
    jj = torch.tensor([1, 0], device="cuda", dtype=torch.long)

    coords, valid_mask = \
        pops.projective_transform(Gs, disps[None], intrinsics[None], ii, jj, return_depth=True)

    depth_jj = 1.0 / disps[jj]

    x = coords[..., 0]
    y = coords[..., 1]

    B, F, H, W = x.shape

    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1

    wx = x - x0.float()
    wy = y - y0.float()

    w00 = (1-wx)*(1-wy)
    w01 = (1-wx)*wy
    w10 = wx*(1-wy)
    w11 = wx*wy

    in_bounds = (
        (x0 >= 0) & (x1 <= W) &
        (y0 >= 0) & (y1 <= H)
    ).unsqueeze(-1)

    dmask = valid_mask.bool() & in_bounds

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
    corr = torch.exp(-K * depth_error) * dmask

    mask = (disps > 0.25 * disps.mean())

    points_np = points[mask].cpu().numpy()

    B, H, W = disps.shape
    colors = torch.zeros((B, H, W, 3), device=disps.device, dtype=torch.float32)

    colormap = plt.colormaps.get_cmap('jet')
    colors_np = colors[mask].cpu().numpy()
    corr_masked = corr[mask[None].unsqueeze(-1)].cpu().numpy()
    jet_rgb = colormap(corr_masked)[:, :3]  # shape (N, 3)

    colors_np[:, :] = jet_rgb

    # --- Create point cloud ---
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_np)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(height=960, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.add_geometry(point_cloud)

    pose_mats = SE3(poses).inv().matrix().cpu().numpy()
    for i in range(len(poses)):
        cam_actor = create_camera_actor(True, cam_scale)
        cam_actor.transform(pose_mats[i])
        vis.add_geometry(cam_actor)

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", type=str, help="path to image directory")
    parser.add_argument("--filter_threshold", type=float, default=0.005)
    parser.add_argument("--filter_count", type=int, default=3)
    parser.add_argument("--cam_scale", type=float, default=0.05)
    args = parser.parse_args()

    view_reconstruction(args.datapath, args.filter_threshold, args.filter_count, args.cam_scale)