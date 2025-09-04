import sys
sys.path.append('droid_slam')

from visualization import create_camera_actor
from plot_trajectory import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtfile")
    parser.add_argument("--evalfile")
    parser.add_argument("--cam_scale", type=float, default=0.05)
    parser.add_argument("--start_scale", type=float, default=0.02)

    args = parser.parse_args()

    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    traj_ref = file_interface.read_tum_trajectory_file(args.gtfile)
    traj_est = file_interface.read_tum_trajectory_file(args.evalfile)

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    print(result)

    geometries = []

    #simulated = red
    for pos,quat in zip(traj_est.positions_xyz, traj_est.orientations_quat_wxyz):
        cam_actor = create_camera_actor(True, args.cam_scale)
        R = cam_actor.get_rotation_matrix_from_quaternion(quat)
        cam_actor.rotate(R, center=(0, 0, 0))
        cam_actor.translate(pos)
        geometries.append(cam_actor)
    
    #ground truth = green/blue
    for pos,quat in zip(traj_ref.positions_xyz, traj_ref.orientations_quat_wxyz):
        cam_actor = create_camera_actor(False, args.cam_scale)
        R = cam_actor.get_rotation_matrix_from_quaternion(quat)
        cam_actor.rotate(R, center=(0, 0, 0))
        cam_actor.translate(pos)
        geometries.append(cam_actor)
    
    if args.start_scale>0.0: #Mark starts
        radius = args.start_scale
        sim_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sim_sphere.paint_uniform_color([1, 0.5, 0.5])
        sim_sphere.translate(traj_est.positions_xyz[0])
        geometries.append(sim_sphere)

        re_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        re_sphere.paint_uniform_color([0.5, 1.0, 0.5])
        re_sphere.translate(traj_ref.positions_xyz[0])
        geometries.append(re_sphere)

    visualize_trajectories([traj_est.positions_xyz, traj_ref.positions_xyz], [[1,0,0],[0,1,0]], geometries)

