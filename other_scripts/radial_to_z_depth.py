# OPENCV_IO_ENABLE_OPENEXR=1

import numpy as np
import cv2
import glob
import argparse
import os

def radial_to_zdepth(depth_radial, fx, fy, cx, cy):
    """
    Convert radial depth map (meters) to Z-depth (optical axis).
    depth_radial: HxW array, radial depth in meters
    fx, fy, cx, cy: intrinsics
    """
    H, W = depth_radial.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    x = (u - cx) / fx
    y = (v - cy) / fy

    norm = np.sqrt(x**2 + y**2 + 1.0)
    depth_z = depth_radial / norm

    return depth_z

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    args = parser.parse_args()

    depth_files = os.listdir(os.path.join(args.datapath,"depth"))

    if not os.path.exists(os.path.join(args.datapath,"zdepth")):
        os.makedirs(os.path.join(args.datapath,"zdepth"))

    fx, fy, cx, cy = np.loadtxt(os.path.join(args.datapath, 'calibration.txt')).tolist()

    for file in depth_files:
        depth_radial = cv2.imread(os.path.join(args.datapath,"depth",file), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 5000.0
        depth_z = radial_to_zdepth(depth_radial, fx, fy, cx, cy)
        cv2.imwrite(os.path.join(args.datapath,"zdepth",file), (depth_z * 5000).astype(np.uint16))
