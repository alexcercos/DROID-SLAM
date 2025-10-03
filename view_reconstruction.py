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

from cuda_timer import CudaTimer

def view_reconstruction(datapath: str, filter_thresh=0.005, filter_count=2, cam_scale=0.05):
    # Load .npy files and convert to torch tensors
    images = torch.from_numpy(np.load(f"{datapath}/images.npy")).cuda()[..., ::2, ::2]
    disps = torch.from_numpy(np.load(f"{datapath}/disps.npy")).cuda()[..., ::2, ::2]
    poses = torch.from_numpy(np.load(f"{datapath}/poses.npy")).cuda()
    intrinsics = 4 * torch.from_numpy(np.load(f"{datapath}/intrinsics.npy")).cuda()

    disps = disps.contiguous()

    index = torch.arange(len(images), device="cuda")
    thresh = filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))

    with CudaTimer("iproj"):
        points = droid_backends.iproj(SE3(poses).inv().data, disps, intrinsics[0])
    colors = images[:,[2,1,0]].permute(0,2,3,1) / 255.0

    with CudaTimer("filter"):
        counts = droid_backends.depth_filter(poses, disps, intrinsics[0], index, thresh)

    mask = (counts >= filter_count) & (disps > .25 * disps.mean())
    points_np = points[mask].cpu().numpy()
    colors_np = colors[mask].cpu().numpy()

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_np)
    point_cloud.colors = o3d.utility.Vector3dVector(colors_np)

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=960, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.add_geometry(point_cloud)

    # get pose matrices as a nx4x4 numpy array
    pose_mats = SE3(poses).inv().matrix().cpu().numpy()

    ### add camera actor ###
    for i in range(len(poses)):
        cam_actor = create_camera_actor(False, cam_scale)
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