
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp

from lietorch import SE3
from .base import RGBDDataset
from .stream import RGBDStream

test_split = ["kitchen_slow", "kitchen_small", "kitchen_table"]

class CustomDataset(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', imgfolder="ir", **kwargs):
        self.mode = mode
        self.n_frames = 2
        self.imgfolder = imgfolder
        super(CustomDataset, self).__init__(name='CustomDataset', **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return scene in test_split

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building CUSTOM dataset")

        scene_info = {}
        scenes = glob.glob(osp.join(self.root, '*'))
        for scene in tqdm(sorted(scenes)):
            print(scene)
            images = sorted(glob.glob(osp.join(scene, f'{self.imgfolder}/*.png')))
            #TODO could add TOF
            depths = sorted(glob.glob(osp.join(scene, 'gtdepth/*.png')))
            
            poses = np.loadtxt(osp.join(scene, 'groundtruth.txt'), delimiter=' ', skiprows=1)
            poses = poses[:, [0, 1, 2, 6, 3, 4, 5]] # --> In theory, XYZ WXYZ
            poses[:,:3] /= CustomDataset.DEPTH_SCALE
            intrinsics = [CustomDataset.calib_read(scene)] * len(images)

            # graph of co-visible frames based on flow
            graph = self.build_frame_graph(poses, depths, intrinsics)

            scene = '/'.join(scene.split('/'))
            scene_info[scene] = {'images': images, 'depths': depths, 
                'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info

    @staticmethod
    def calib_read(scene):
        return np.loadtxt(osp.join(scene, 'calibration.txt'), delimiter=' ', max_rows=1)

    @staticmethod
    def image_read(image_file):
        image = cv2.imread(image_file)
#        h0, w0, _ = image.shape
#        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
#        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        image = cv2.resize(image, (800, 800))
        return image

    @staticmethod
    def depth_read(depth_file):

        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) / 5000.0 / CustomDataset.DEPTH_SCALE
#        h0, w0 = depth.shape
#        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
#        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        depth = cv2.resize(depth, (800, 800))
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        depth[depth < 0.01] = np.mean(depth)
        return depth

#Possibly unused (Tartanair equivalents?

# class CustomDatasetStream(RGBDStream):
#     def __init__(self, datapath, **kwargs):
#         super(CustomDatasetStream, self).__init__(datapath=datapath, **kwargs)

#     def _build_dataset_index(self):
#         """ build list of images, poses, depths, and intrinsics """
#         self.root = 'datasets/CustomDataset'

#         scene = osp.join(self.root, self.datapath)
#         image_glob = osp.join(scene, 'image_left/*.png') #TODO modify
#         images = sorted(glob.glob(image_glob))

#         poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ') #TODO modify
#         poses = poses[:, [0, 1, 2, 6, 3, 4, 5]] # --> In theory, XYZ WXYZ

#         poses = SE3(torch.as_tensor(poses))
#         poses = poses[[0]].inv() * poses
#         poses = poses.data.cpu().numpy()

#         intrinsic = self.calib_read(self.datapath)
#         intrinsics = np.tile(intrinsic[None], (len(images), 1))

#         self.images = images[::int(self.frame_rate)]
#         self.poses = poses[::int(self.frame_rate)]
#         self.intrinsics = intrinsics[::int(self.frame_rate)]

#     @staticmethod
#     def calib_read(datapath):
#         return np.array([320.0, 320.0, 320.0, 240.0]) #TODO modify

#     @staticmethod
#     def image_read(image_file):
#         return cv2.imread(image_file)


# class CustomDatasetTestStream(RGBDStream):
#     def __init__(self, datapath, **kwargs):
#         super(CustomDatasetTestStream, self).__init__(datapath=datapath, **kwargs)

#     def _build_dataset_index(self):
#         """ build list of images, poses, depths, and intrinsics """
#         self.root = 'datasets/mono'
#         image_glob = osp.join(self.root, self.datapath, '*.png') #TODO modify
#         images = sorted(glob.glob(image_glob))

#         poses = np.loadtxt(osp.join(self.root, 'mono_gt', self.datapath + '.txt'), delimiter=' ') #TODO modify
#         poses = poses[:, [0, 1, 2, 6, 3, 4, 5]] # --> In theory, XYZ WXYZ

#         poses = SE3(torch.as_tensor(poses))
#         poses = poses[[0]].inv() * poses
#         poses = poses.data.cpu().numpy()

#         intrinsic = self.calib_read(self.datapath)
#         intrinsics = np.tile(intrinsic[None], (len(images), 1))

#         self.images = images[::int(self.frame_rate)]
#         self.poses = poses[::int(self.frame_rate)]
#         self.intrinsics = intrinsics[::int(self.frame_rate)]

#     @staticmethod
#     def calib_read(datapath):
#         return np.array([320.0, 320.0, 320.0, 240.0]) #TODO modify

#     @staticmethod
#     def image_read(image_file):
#         return cv2.imread(image_file)
