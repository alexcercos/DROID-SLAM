import sys
sys.path.append('droid_slam')

import numpy as np
import open3d as o3d

import os
from visualization import create_camera_actor
import argparse


def visualize_trajectories(arr_positions, arr_colors=[[1,0,0]], geometries = []):
    # Create Open3D LineSet

    for positions,color in zip(arr_positions,arr_colors):

        points = o3d.utility.Vector3dVector(positions)
        lines = [[i, i + 1] for i in range(len(positions) - 1)]
        line_set = o3d.geometry.LineSet()
        line_set.points = points
        line_set.lines = o3d.utility.Vector2iVector(lines)

        colors = [color for _ in lines]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        geometries.append(line_set)

    # Add a coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    geometries.append(coord_frame)

    o3d.visualization.draw_geometries(geometries,
                                      zoom=0.8,
                                      front=[0.0, -1.0, 0.0],
                                      lookat=[0.0, 0.0, 0.0],
                                      up=[0.0, 0.0, 1.0])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    parser.add_argument("--cam_scale", type=float, default=0.05)

    args = parser.parse_args()

    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface

    gt_file = os.path.join(args.datapath, 'groundtruth.txt')
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)


    geometries = []
    
    #ground truth = green/blue
    for pos,quat in zip(traj_ref.positions_xyz, traj_ref.orientations_quat_wxyz):
        cam_actor = create_camera_actor(False, args.cam_scale)
        R = cam_actor.get_rotation_matrix_from_quaternion(quat)
        cam_actor.rotate(R, center=(0, 0, 0))
        cam_actor.translate(pos)
        geometries.append(cam_actor)
    
    visualize_trajectories([traj_ref.positions_xyz], [[0,1,0]], geometries)

