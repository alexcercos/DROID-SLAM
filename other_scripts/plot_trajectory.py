import numpy as np
import open3d as o3d

def load_trajectory(filename):
    positions = []

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            tx, ty, tz = map(float, parts[1:4])
            positions.append([tx, ty, tz])

    return np.array(positions)

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

if __name__ == "__main__":
    # Replace with your actual filename
    trajectory_file = "/home/alejandro/Documentos/DROID-SLAM/datasets/ETH3D-SLAM/training/" + input("Sequence: ") + "/groundtruth.txt"
    trajectory = load_trajectory(trajectory_file)
    visualize_trajectories([trajectory])
