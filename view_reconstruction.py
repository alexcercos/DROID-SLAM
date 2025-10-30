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

def view_reconstruction(datapath: str,
                        filter_thresh=0.005,
                        filter_count=2,
                        cam_scale=0.05,
                        rotate=False,
                        show_axis=False,
                        lookat_point=(0.0, 0.0, 0.0),
                        world_rotation_xyz=(0.0,0.0,0.0),
                        rotate_radius=3.0,
                        rotate_height=1.0,
                        rotate_speed_deg=1.0,
                        save_frames=False,
                        save_every=10,
                        save_dir="frames"):
    # Load .npy files and convert to torch tensors
    images = torch.from_numpy(np.load(f"{datapath}/images.npy")).cuda()[..., ::2, ::2]
    disps = torch.from_numpy(np.load(f"{datapath}/disps.npy")).cuda()[..., ::2, ::2]
    poses = torch.from_numpy(np.load(f"{datapath}/poses.npy")).cuda()
    intrinsics = 4 * torch.from_numpy(np.load(f"{datapath}/intrinsics.npy")).cuda()

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    world_rotation_xyz = (math.radians(world_rotation_xyz[0]),math.radians(world_rotation_xyz[1]),math.radians(world_rotation_xyz[2]))

    T = np.eye(4)
    T[:3, :3] = coord_frame.get_rotation_matrix_from_xyz(world_rotation_xyz)

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
    point_cloud.transform(T)

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
        cam_actor.transform(T)
        vis.add_geometry(cam_actor)
    
    if show_axis:
        vis.add_geometry(coord_frame)
        coord_frame2 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        coord_frame2.transform(T)
        vis.add_geometry(coord_frame2)

    #------------------

    if save_frames:
        os.makedirs(save_dir, exist_ok=True)
        frame_idx = {'count': 0}

    # --- Rotation state ---
    state = {
        'rotating': rotate,
        'angle': 0.0
    }

    def toggle_rotation(vis):
        state['rotating'] = not state['rotating']
        print(f"{'Started' if state['rotating'] else 'Stopped'} rotation.")
        return False

    vis.register_key_callback(ord('R'), toggle_rotation)

    # --- Camera rotation callback ---
    def rotate_and_capture(vis):
        if not state['rotating']:
            return False

        ctr = vis.get_view_control()
        cam = ctr.convert_to_pinhole_camera_parameters()

        # Increase orbit angle
        state['angle'] += math.radians(rotate_speed_deg)
        theta = state['angle']

        # ----- User-configurable orbit path -----
        # Y is the UP axis in this convention.
        # Orbit on the XZ-plane around the chosen look-at point.
        lookat = np.array(lookat_point, dtype=float)  # <-- defined outside, see below
        up = np.array([0.0, 1.0, 0.0])               # world-up along Y

        # Camera orbit position
        x = rotate_radius * math.cos(theta)
        z = rotate_radius * math.sin(theta)
        y = rotate_height
        cam_pos = lookat + np.array([x, y, z])

        # ----- Build orientation -----
        forward = (lookat - cam_pos)
        forward /= np.linalg.norm(forward)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up_corrected = np.cross(forward, right)
        R = np.stack([right, up_corrected, forward], axis=1)

        # Build new extrinsic (camera-to-world inverse)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R.T
        extrinsic[:3, 3] = -R.T @ cam_pos
        cam.extrinsic = extrinsic

        ctr.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)

        # Optional frame saving
        if save_frames and state['angle']<2*math.pi:

            frame_idx['count'] += 1
            if frame_idx['count'] % save_every == 0:
                fname = os.path.join(save_dir, f"frame_{frame_idx['count']:06d}.png")
                vis.capture_screen_image(fname, do_render=True)
                print(f"[Saved] {fname}")

        vis.update_renderer()
        return False

    vis.register_animation_callback(rotate_and_capture)

    #------------------

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", type=str, help="path to image directory")
    parser.add_argument("--filter_threshold", type=float, default=0.005)
    parser.add_argument("--filter_count", type=int, default=3)
    parser.add_argument("--cam_scale", type=float, default=0.03)
    args = parser.parse_args()

    view_reconstruction(args.datapath, 
                        args.filter_threshold, 
                        args.filter_count, 
                        args.cam_scale,
                        rotate_speed_deg=3.0,
                        rotate_height=-1,
                        rotate_radius=1.5,
                        lookat_point=(-3, -1, 1),
                        world_rotation_xyz=(0, 0, 0),
                        save_frames=True,
                        save_every=1,
                        # show_axis=True
                        )
