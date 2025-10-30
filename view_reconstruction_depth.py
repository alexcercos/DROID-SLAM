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

def view_reconstruction(datapath: str, filter_thresh=0.005, filter_count=2, cam_scale=0.05,
                        depth_min=0.0,
                        depth_max=0.5):
    # Load .npy files and convert to torch tensors
    images = torch.from_numpy(np.load(f"{datapath}/images.npy")).cuda()[..., ::2, ::2]
    disps = torch.from_numpy(np.load(f"{datapath}/disps.npy")).cuda()[..., ::2, ::2]
    poses = torch.from_numpy(np.load(f"{datapath}/poses.npy")).cuda()
    intrinsics = 4 * torch.from_numpy(np.load(f"{datapath}/intrinsics.npy")).cuda()

    disps = disps.contiguous()

    index = torch.arange(len(images), device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1, 2]))

    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
    
    # Get current camera position (first camera)
    current_cam_pose = SE3(poses[0]).inv().matrix().cpu().numpy()
    cam_center = current_cam_pose[:3, 3]

    with CudaTimer("filter"):
        counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    mask = (counts >= filter_count) & (disps > .25 * disps.mean())
    points_np = points[mask].cpu().numpy()

    # --- Inverse depth coloring ---
    depths = np.linalg.norm(points_np - cam_center[None, :], axis=1)
    inv_depths = 1.0 / np.clip(depths, 1e-6, None)

    vmin = depth_min if depth_min is not None else np.percentile(inv_depths, 1)
    vmax = depth_max if depth_max is not None else np.percentile(inv_depths, 99)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.colormaps.get_cmap('jet')

    colors_np = colormap(norm(inv_depths))[:, :3]

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

    def recolor(vis):
        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        extrinsic = np.linalg.inv(cam_params.extrinsic)  # world-from-camera
        cam_center = extrinsic[:3, 3]

        depths = np.linalg.norm(points_np - cam_center[None, :], axis=1)
        inv_depths = 1.0 / np.clip(depths, 1e-6, None)
        print(np.min(inv_depths),np.max(inv_depths))

        vmin = depth_min if depth_min is not None else np.percentile(inv_depths, 1)
        vmax = depth_max if depth_max is not None else np.percentile(inv_depths, 99)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        colors = colormap(norm(inv_depths))[:, :3]
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(point_cloud)
        vis.update_renderer()
        print(f"Recolored using inverse depth: inv_depth [{vmin:.4f}, {vmax:.4f}]")

    # Press 'C' to recolor based on current camera position
    vis.register_key_callback(ord('C'), recolor)

    print("Press 'C' to recolor points based on inverse depth from current camera position.")
    print("Close the window to exit.")

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", type=str, help="path to image directory")
    parser.add_argument("--filter_threshold", type=float, default=0.005)
    parser.add_argument("--filter_count", type=int, default=3)
    parser.add_argument("--cam_scale", type=float, default=0.05)
    parser.add_argument("--dmax", type=float, default=0.5)
    args = parser.parse_args()

    view_reconstruction(args.datapath, args.filter_threshold, args.filter_count, args.cam_scale, depth_max=args.dmax)