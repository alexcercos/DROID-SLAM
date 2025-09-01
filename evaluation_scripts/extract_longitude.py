import sys
sys.path.append('droid_slam')


import os
import glob 
import math
import argparse

from droid import Droid


imagefolder = "rgb"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath")
    args = parser.parse_args()

    ### run evaluation ###

    from evo.tools import file_interface
    from evo.core import sync
    import evo.main_ape as main_ape
    from evo.core.metrics import PoseRelation

    image_path = os.path.join(args.datapath, imagefolder)
    images_list = sorted(glob.glob(os.path.join(image_path, '*.png')))
    tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]

    gt_file = os.path.join(args.datapath, 'groundtruth.txt')
    traj_ref = file_interface.read_tum_trajectory_file(gt_file)

    total_dist = 0
    last_pos = None
    for ts,pos in zip(traj_ref.timestamps,traj_ref.positions_xyz):
        if ts<tstamps[0]: #first ts
            continue
        if ts>tstamps[-1]: #last ts
            continue

        if last_pos is not None:
            total_dist += math.sqrt((pos[0]-last_pos[0])**2+(pos[1]-last_pos[1])**2+(pos[2]-last_pos[2])**2)
        last_pos = pos
    
    print(args.datapath,total_dist)